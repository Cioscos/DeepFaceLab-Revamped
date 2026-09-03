import json
import multiprocessing
import os
import re
import sys
import threading
import time
import types

import colorama
import cv2
import numpy as np
from tqdm import tqdm

from core import stdex

try:
    import IPython #if success we are in colab
    from IPython.display import display, clear_output
    import PIL
    import matplotlib.pyplot as plt
    is_colab = True
except:
    is_colab = False

yn_str = {True:'y',False:'n'}

_ANSWERS_MISSING = object()

def prompt_key(text):
    # Normalized key of a prompt: lowercase, characters outside
    # [a-z0-9_ -] dropped, whitespace runs become '-'. Pre-supplied
    # answers (DFL_ANSWERS_FILE) are looked up by this key.
    s = text.lower()
    s = re.sub(r"[^a-z0-9_\s-]", "", s)
    return re.sub(r"\s+", "-", s.strip())

class ProgressLog:
    """Il quinto canale: le barre di avanzamento, come righe JSON.

    Nasce solo se DFL_PROGRESS_FILE c'e' E se siamo il processo che
    l'utente ha lanciato: i pool nascono con spawn ed ereditano
    l'ambiente, e otto figli sullo stesso file farebbero danzare la barra
    all'indietro.
    """

    MIN_INTERVALLO = 0.5

    def __init__(self, path, clock=time.time):
        self.path = path
        self._clock = clock
        self._id = 0
        self._aperta = False
        self._n = 0
        self._ultima = 0.0

    @staticmethod
    def da_ambiente(environ=None, nome_processo=None, clock=time.time):
        environ = os.environ if environ is None else environ
        path = environ.get("DFL_PROGRESS_FILE")
        if not path:
            return None
        nome = nome_processo or multiprocessing.current_process().name
        if nome != "MainProcess":
            return None
        return ProgressLog(path, clock=clock)

    def apri(self, desc, total, initial=0):
        if self._aperta:
            self.chiudi()
        self._id += 1
        self._aperta = True
        self._n = initial or 0
        self._scrivi({"op": "open", "id": self._id, "desc": str(desc or ""),
                      "total": total, "initial": self._n})
        self._ultima = self._clock()

    def inc(self, c):
        if not self._aperta:
            return
        self._n += c
        ora = self._clock()
        if ora - self._ultima < self.MIN_INTERVALLO:
            return
        self._ultima = ora
        self._scrivi({"op": "inc", "id": self._id, "n": self._n})

    def chiudi(self):
        if not self._aperta:
            return
        self._scrivi({"op": "inc", "id": self._id, "n": self._n})
        self._scrivi({"op": "close", "id": self._id})
        self._aperta = False

    def _scrivi(self, payload):
        # allow_nan=False: un NaN solleva qui, dove costa una riga persa,
        # invece di attraversare il canale e morire dentro un paintEvent.
        try:
            riga = json.dumps(payload, allow_nan=False)
        except ValueError:
            return
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(riga + "\n")
        except OSError:
            pass

class InteractBase(object):
    EVENT_LBUTTONDOWN = 1
    EVENT_LBUTTONUP = 2
    EVENT_MBUTTONDOWN = 3
    EVENT_MBUTTONUP = 4
    EVENT_RBUTTONDOWN = 5
    EVENT_RBUTTONUP = 6
    EVENT_MOUSEWHEEL = 10

    def __init__(self):
        self.named_windows = {}
        self.capture_mouse_windows = {}
        self.capture_keys_windows = {}
        self.mouse_events = {}
        self.key_events = {}
        self.pg_bar = None
        self.pg_log = ProgressLog.da_ambiente()
        self.focus_wnd_name = None
        self.error_log_line_prefix = '/!\\ '

        self.process_messages_callbacks = {}

    def is_support_windows(self):
        return False

    def is_colab(self):
        return False

    def _preset_answers(self):
        # None = interactive mode; dict = answers pre-supplied by an
        # external driver. Reloaded when the file path changes.
        path = os.environ.get('DFL_ANSWERS_FILE')
        if path is None:
            return None
        if getattr(self, '_answers_path', None) != path:
            with open(path, 'r', encoding='utf-8') as f:
                self._answers = { prompt_key(k): v for k, v in json.load(f).items() }
            self._answers_path = path
        return self._answers

    def _preset(self, s):
        # (active, value): value is _ANSWERS_MISSING when the key is absent.
        answers = self._preset_answers()
        if answers is None:
            return False, _ANSWERS_MISSING
        return True, answers.get(prompt_key(s), _ANSWERS_MISSING)

    def on_destroy_all_windows(self):
        raise NotImplemented

    def on_create_window (self, wnd_name):
        raise NotImplemented

    def on_destroy_window (self, wnd_name):
        raise NotImplemented

    def on_show_image (self, wnd_name, img):
        raise NotImplemented

    def on_capture_mouse (self, wnd_name):
        raise NotImplemented

    def on_capture_keys (self, wnd_name):
        raise NotImplemented

    def on_process_messages(self, sleep_time=0):
        raise NotImplemented

    def on_wait_any_key(self):
        raise NotImplemented

    def log_info(self, msg, end='\n'):
        if self.pg_bar is not None:
            print ("\n")
        print (msg, end=end)

    def log_err(self, msg, end='\n'):
        if self.pg_bar is not None:
            print ("\n")
        print (f'{self.error_log_line_prefix}{msg}', end=end)

    def named_window(self, wnd_name):
        if wnd_name not in self.named_windows:
            #we will show window only on first show_image
            self.named_windows[wnd_name] = 0
            self.focus_wnd_name = wnd_name
        else: print("named_window: ", wnd_name, " already created.")

    def destroy_all_windows(self):
        if len( self.named_windows ) != 0:
            self.on_destroy_all_windows()
            self.named_windows = {}
            self.capture_mouse_windows = {}
            self.capture_keys_windows = {}
            self.mouse_events = {}
            self.key_events = {}
            self.focus_wnd_name = None

    def destroy_window(self, wnd_name):
        if wnd_name in self.named_windows:
            self.on_destroy_window(wnd_name)
            self.named_windows.pop(wnd_name)

            if wnd_name == self.focus_wnd_name:
                self.focus_wnd_name = list(self.named_windows.keys())[-1] if len( self.named_windows ) != 0 else None

            if wnd_name in self.capture_mouse_windows:
                self.capture_mouse_windows.pop(wnd_name)

            if wnd_name in self.capture_keys_windows:
                self.capture_keys_windows.pop(wnd_name)

            if wnd_name in self.mouse_events:
                self.mouse_events.pop(wnd_name)

            if wnd_name in self.key_events:
                self.key_events.pop(wnd_name)

    def show_image(self, wnd_name, img):
        if wnd_name in self.named_windows:
            if self.named_windows[wnd_name] == 0:
                self.named_windows[wnd_name] = 1
                self.on_create_window(wnd_name)
                if wnd_name in self.capture_mouse_windows:
                    self.capture_mouse(wnd_name)
            self.on_show_image(wnd_name,img)
        else: print("show_image: named_window ", wnd_name, " not found.")

    def capture_mouse(self, wnd_name):
        if wnd_name in self.named_windows:
            self.capture_mouse_windows[wnd_name] = True
            if self.named_windows[wnd_name] == 1:
                self.on_capture_mouse(wnd_name)
        else: print("capture_mouse: named_window ", wnd_name, " not found.")

    def capture_keys(self, wnd_name):
        if wnd_name in self.named_windows:
            if wnd_name not in self.capture_keys_windows:
                self.capture_keys_windows[wnd_name] = True
                self.on_capture_keys(wnd_name)
            else: print("capture_keys: already set for window ", wnd_name)
        else: print("capture_keys: named_window ", wnd_name, " not found.")

    def progress_bar(self, desc, total, leave=True, initial=0):
        if self.pg_bar is None:
            self.pg_bar = tqdm( total=total, desc=desc, leave=leave, ascii=True, initial=initial )
            if self.pg_log is not None:
                self.pg_log.apri(desc, total, initial)
        else: print("progress_bar: already set.")

    def progress_bar_inc(self, c):
        if self.pg_bar is not None:
            self.pg_bar.n += c
            self.pg_bar.refresh()
            if self.pg_log is not None:
                self.pg_log.inc(c)
        else: print("progress_bar not set.")

    def progress_bar_close(self):
        if self.pg_bar is not None:
            self.pg_bar.close()
            self.pg_bar = None
            if self.pg_log is not None:
                self.pg_log.chiudi()
        else: print("progress_bar not set.")

    def progress_bar_generator(self, data, desc=None, leave=True, initial=0):
        self.pg_bar = tqdm( data, desc=desc, leave=leave, ascii=True, initial=initial )
        if self.pg_log is not None:
            self.pg_log.apri(desc, getattr(self.pg_bar, "total", None), initial)
        for x in self.pg_bar:
            yield x
            if self.pg_log is not None:
                self.pg_log.inc(1)
        self.pg_bar.close()
        self.pg_bar = None
        if self.pg_log is not None:
            self.pg_log.chiudi()

    def add_process_messages_callback(self, func ):
        tid = threading.get_ident()
        callbacks = self.process_messages_callbacks.get(tid, None)
        if callbacks is None:
            callbacks = []
            self.process_messages_callbacks[tid] = callbacks

        callbacks.append ( func )

    def process_messages(self, sleep_time=0):
        callbacks = self.process_messages_callbacks.get(threading.get_ident(), None)
        if callbacks is not None:
            for func in callbacks:
                func()

        self.on_process_messages(sleep_time)

    def wait_any_key(self):
        self.on_wait_any_key()

    def add_mouse_event(self, wnd_name, x, y, ev, flags):
        if wnd_name not in self.mouse_events:
            self.mouse_events[wnd_name] = []
        self.mouse_events[wnd_name] += [ (x, y, ev, flags) ]

    def add_key_event(self, wnd_name, ord_key, ctrl_pressed, alt_pressed, shift_pressed):
        if wnd_name not in self.key_events:
            self.key_events[wnd_name] = []
        self.key_events[wnd_name] += [ (ord_key, chr(ord_key) if ord_key <= 255 else chr(0), ctrl_pressed, alt_pressed, shift_pressed) ]

    def get_mouse_events(self, wnd_name):
        ar = self.mouse_events.get(wnd_name, [])
        self.mouse_events[wnd_name] = []
        return ar

    def get_key_events(self, wnd_name):
        ar = self.key_events.get(wnd_name, [])
        self.key_events[wnd_name] = []
        return ar

    def input(self, s):
        return input(s)

    def input_number(self, s, default_value, valid_list=None, show_default_value=True, add_info=None, help_message=None):
        active, v = self._preset(s)
        if active:
            if v is _ANSWERS_MISSING:
                result = default_value
            else:
                result = float(v)
                if (valid_list is not None) and (result not in valid_list):
                    result = default_value
            self.log_info("%s : %s" % (s, result))
            return result

        if show_default_value and default_value is not None:
            s = f"[{default_value}] {s}"

        if add_info is not None or \
           help_message is not None:
            s += " ("

        if add_info is not None:
            s += f" {add_info}"
        if help_message is not None:
            s += " ?:help"

        if add_info is not None or \
           help_message is not None:
            s += " )"

        s += " : "

        while True:
            try:
                inp = input(s)
                if len(inp) == 0:
                    result = default_value
                    break

                if help_message is not None and inp == '?':
                    print (help_message)
                    continue

                i = float(inp)
                if (valid_list is not None) and (i not in valid_list):
                    result = default_value
                    break
                result = i
                break
            except:
                result = default_value
                break

        print(result)
        return result

    def input_int(self, s, default_value, valid_range=None, valid_list=None, add_info=None, show_default_value=True, help_message=None):
        active, v = self._preset(s)
        if active:
            if v is _ANSWERS_MISSING:
                result = default_value
            else:
                result = int(v)
                if valid_range is not None:
                    result = int(np.clip(result, valid_range[0], valid_range[1]))
                if (valid_list is not None) and (result not in valid_list):
                    result = default_value
            self.log_info("%s : %s" % (s, result))
            return result

        if show_default_value:
            if len(s) != 0:
                s = f"[{default_value}] {s}"
            else:
                s = f"[{default_value}]"

        if add_info is not None or \
           valid_range is not None or \
           help_message is not None:
            s += " ("

        if valid_range is not None:
            s += f" {valid_range[0]}-{valid_range[1]}"

        if add_info is not None:
            s += f" {add_info}"

        if help_message is not None:
            s += " ?:help"

        if add_info is not None or \
           valid_range is not None or \
           help_message is not None:
            s += " )"

        s += " : "

        while True:
            try:
                inp = input(s)
                if len(inp) == 0:
                    raise ValueError("")

                if help_message is not None and inp == '?':
                    print (help_message)
                    continue

                i = int(inp)
                if valid_range is not None:
                    i = int(np.clip(i, valid_range[0], valid_range[1]))

                if (valid_list is not None) and (i not in valid_list):
                    i = default_value

                result = i
                break
            except:
                result = default_value
                break
        print (result)
        return result

    def input_bool(self, s, default_value, help_message=None):
        active, v = self._preset(s)
        if active:
            result = default_value if v is _ANSWERS_MISSING else bool(v)
            self.log_info("%s : %s" % (s, "y" if result else "n"))
            return result

        s = f"[{yn_str[default_value]}] {s} ( y/n"

        if help_message is not None:
            s += " ?:help"
        s += " ) : "

        while True:
            try:
                inp = input(s)
                if len(inp) == 0:
                    raise ValueError("")

                if help_message is not None and inp == '?':
                    print (help_message)
                    continue

                return bool ( {"y":True,"n":False}.get(inp.lower(), default_value) )
            except:
                print ( "y" if default_value else "n" )
                return default_value

    def input_str(self, s, default_value=None, valid_list=None, show_default_value=True, help_message=None):
        active, v = self._preset(s)
        if active:
            if v is _ANSWERS_MISSING:
                result = default_value
            else:
                result = str(v)
                if valid_list is not None and result not in valid_list:
                    if result.lower() in valid_list:
                        result = result.lower()
                    else:
                        result = default_value
            self.log_info("%s : %s" % (s, result))
            return result

        if show_default_value and default_value is not None:
            s = f"[{default_value}] {s}"

        if valid_list is not None or \
           help_message is not None:
            s += " ("

        if valid_list is not None:
            s += " " + "/".join(valid_list)

        if help_message is not None:
            s += " ?:help"

        if valid_list is not None or \
           help_message is not None:
            s += " )"

        s += " : "


        while True:
            try:
                inp = input(s)

                if len(inp) == 0:
                    if default_value is None:
                        print("")
                        return None
                    result = default_value
                    break

                if help_message is not None and inp == '?':
                    print(help_message)
                    continue

                if valid_list is not None:
                    if inp.lower() in valid_list:
                        result = inp.lower()
                        break
                    if inp in valid_list:
                        result = inp
                        break
                    continue

                result = inp
                break
            except:
                result = default_value
                break

        print(result)
        return result

    def input_process(self, stdin_fd, sq, str):
        sys.stdin = os.fdopen(stdin_fd)
        try:
            inp = input (str)
            sq.put (True)
        except:
            sq.put (False)

    def input_in_time (self, str, max_time_sec):
        if self._preset_answers() is not None:
            return True

        sq = multiprocessing.Queue()
        p = multiprocessing.Process(target=self.input_process, args=( sys.stdin.fileno(), sq, str))
        p.daemon = True
        p.start()
        t = time.time()
        inp = False
        while True:
            if not sq.empty():
                inp = sq.get()
                break
            if time.time() - t > max_time_sec:
                break


        p.terminate()
        p.join()

        old_stdin = sys.stdin
        sys.stdin = os.fdopen( os.dup(sys.stdin.fileno()) )
        old_stdin.close()
        return inp

    def input_process_skip_pending(self, stdin_fd):
        sys.stdin = os.fdopen(stdin_fd)
        while True:
            try:
                if sys.stdin.isatty():
                    sys.stdin.read()
            except:
                pass

    def input_skip_pending(self):
        if is_colab:
            # currently it does not work on Colab
            return
        if self._preset_answers() is not None:
            # Answers mode: nobody is typing, so there is nothing pending to
            # skip -- and the stdin surgery below is pure damage. It spawns a
            # process holding fd 0, kills it half a second later, and rebinds
            # sys.stdin with os.fdopen(fd) (no dup, so a second owner of the
            # same descriptor). On Windows that leaves the GUI's command pipe
            # at EOF: measured on the merge service, which read its first
            # command at 0.0 s and saw the pipe close at 0.5 s -- exactly this
            # sleep -- between the GPU prompt and "Initializing models", then
            # shut itself down as if the GUI had died. `input_in_time` above
            # already carries the same guard.
            return
        """
        skips unnecessary inputs between the dialogs
        """
        p = multiprocessing.Process(target=self.input_process_skip_pending, args=( sys.stdin.fileno(), ))
        p.daemon = True
        p.start()
        time.sleep(0.5)
        p.terminate()
        p.join()
        sys.stdin = os.fdopen( sys.stdin.fileno() )


class InteractDesktop(InteractBase):
    def __init__(self):
        colorama.init()
        super().__init__()

    def color_red(self):
        pass


    def is_support_windows(self):
        return True

    def on_destroy_all_windows(self):
        cv2.destroyAllWindows()

    def on_create_window (self, wnd_name):
        cv2.namedWindow(wnd_name)

    def on_destroy_window (self, wnd_name):
        cv2.destroyWindow(wnd_name)

    def on_show_image (self, wnd_name, img):
        cv2.imshow (wnd_name, img)

    def on_capture_mouse (self, wnd_name):
        self.last_xy = (0,0)

        def onMouse(event, x, y, flags, param):
            (inst, wnd_name) = param
            if event == cv2.EVENT_LBUTTONDOWN: ev = InteractBase.EVENT_LBUTTONDOWN
            elif event == cv2.EVENT_LBUTTONUP: ev = InteractBase.EVENT_LBUTTONUP
            elif event == cv2.EVENT_RBUTTONDOWN: ev = InteractBase.EVENT_RBUTTONDOWN
            elif event == cv2.EVENT_RBUTTONUP: ev = InteractBase.EVENT_RBUTTONUP
            elif event == cv2.EVENT_MBUTTONDOWN: ev = InteractBase.EVENT_MBUTTONDOWN
            elif event == cv2.EVENT_MBUTTONUP: ev = InteractBase.EVENT_MBUTTONUP
            elif event == cv2.EVENT_MOUSEWHEEL:
                ev = InteractBase.EVENT_MOUSEWHEEL
                x,y = self.last_xy #fix opencv bug when window size more than screen size
            else: ev = 0

            self.last_xy = (x,y)
            inst.add_mouse_event (wnd_name, x, y, ev, flags)
        cv2.setMouseCallback(wnd_name, onMouse, (self,wnd_name) )

    def on_capture_keys (self, wnd_name):
        pass

    def on_process_messages(self, sleep_time=0):

        has_windows = False
        has_capture_keys = False

        if len(self.named_windows) != 0:
            has_windows = True

        if len(self.capture_keys_windows) != 0:
            has_capture_keys = True

        if has_windows or has_capture_keys:
            wait_key_time = max(1, int(sleep_time*1000) )
            ord_key = cv2.waitKeyEx(wait_key_time)
            
            shift_pressed = False
            if ord_key != -1:
                chr_key = chr(ord_key) if ord_key <= 255 else chr(0)

                if chr_key >= 'A' and chr_key <= 'Z':
                    shift_pressed = True
                    ord_key += 32
                elif chr_key == '?':
                    shift_pressed = True
                    ord_key = ord('/')
                elif chr_key == '<':
                    shift_pressed = True
                    ord_key = ord(',')
                elif chr_key == '>':
                    shift_pressed = True
                    ord_key = ord('.')
        else:
            if sleep_time != 0:
                time.sleep(sleep_time)

        if has_capture_keys and ord_key != -1:
            self.add_key_event ( self.focus_wnd_name, ord_key, False, False, shift_pressed)

    def on_wait_any_key(self):
        cv2.waitKey(0)

class InteractColab(InteractBase):

    def is_support_windows(self):
        return False

    def is_colab(self):
        return True

    def on_destroy_all_windows(self):
        pass
        #clear_output()

    def on_create_window (self, wnd_name):
        pass
        #clear_output()

    def on_destroy_window (self, wnd_name):
        pass

    def on_show_image (self, wnd_name, img):
        pass
        # # cv2 stores colors as BGR; convert to RGB
        # if img.ndim == 3:
        #     if img.shape[2] == 4:
        #         img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        #     else:
        #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = PIL.Image.fromarray(img)
        # plt.imshow(img)
        # plt.show()

    def on_capture_mouse (self, wnd_name):
        pass
        #print("on_capture_mouse(): Colab does not support")

    def on_capture_keys (self, wnd_name):
        pass
        #print("on_capture_keys(): Colab does not support")

    def on_process_messages(self, sleep_time=0):
        time.sleep(sleep_time)

    def on_wait_any_key(self):
        pass
        #print("on_wait_any_key(): Colab does not support")

if is_colab:
    interact = InteractColab()
else:
    interact = InteractDesktop()
