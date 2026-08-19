# Third-party model licenses

Some of the `.npy` weight files in this folder were converted from checkpoints
published by other projects. Those projects ask for their copyright notice to
be distributed **with the material**, and the weights ship as a release
archive and not only as source code, so this notice ships inside `facelib/`
next to the files it describes.

Converting a checkpoint to the on-disk format used here changes the container,
not the model: the numbers are the ones the original authors trained, and the
terms below apply to them.

Every section below names the weight files it covers. This file makes no
statement about any file it does not name.

---

## RetinaFace-R50 — `RetinaFaceR50.npy`

* Origin project: <https://github.com/ternaus/retinaface>
* Author: Vladimir Iglovikov
* License: MIT
* Source checkpoint: `retinaface_resnet50_2020-07-20.pth`, from release `0.01`
  of that project.

```
MIT License

Copyright (c) 2020 Vladimir Iglovikov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## PIPNet-68 — `PIPNet68.npy`

* Origin project: <https://github.com/jhb86253817/PIPNet>
* Author: Haibo Jin
* License: MIT
* Source checkpoint: `data_300W/pip_32_16_60_r18_l2_l1_10_1_nb10/epoch59.pth`,
  obtained through the `assale02/PIPNet` mirror on Hugging Face.
* `PIPNet68.npy` also embeds the neighbour index derived from
  `data/data_300W/meanface.txt` of the same project.

```
MIT License

Copyright (c) 2020 Haibo Jin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## FAN 2D and 3D landmarks — `2DFAN.npy`, `3DFAN.npy`

* Origin project: <https://github.com/1adrianb/face-alignment>
* Author: Adrian Bulat
* License: BSD 3-Clause
* `facelib/FANExtractor.py` records that its implementation is ported from
  that project; these weights are the converted form of the checkpoints
  published there.

The BSD 3-Clause license, like MIT, requires the notice below to be reproduced
in a redistribution in binary form, which is what a weight file is.

```
BSD 3-Clause License

Copyright (c) 2017, Adrian Bulat
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

---

## S3FD detector — `S3FD.npy`

* Model: S3FD, the single-shot scale-invariant face detector.
* License: **not stated, and the blank is deliberate.** Every other section
  above carries this line because the origin of its weight file is documented.
  For this one it is not, and what is known — and what is not — is written out
  below instead of being compressed into a single word.

**Where this weight file came from is not recorded.** No comment and no
document in this repository states the origin of `S3FD.npy`, and it is not
reconstructible from the code:
`grep -riE "licen|copyright|http" facelib/S3FDExtractor.py` prints nothing.

**What the code shows is the architecture, which is a different thing.**
`facelib/S3FDExtractor.py` rebuilds the S3FD *architecture*: the layer names
it uses (`conv3_3_norm_mbox_conf`, `fc7_mbox_loc`, `conv6_2`, `conv7_2`) and
the **4** confidence channels on the first detection scale where every later
scale has 2 — the max-out background label of that detector — identify the
**model**. They do not identify the file. Those names are the keys inside
`S3FD.npy` itself: the loader matches each parameter to the file by key and
fails loudly when one is missing, so the code has to spell them exactly as the
weight file does, whatever the file's origin. Nor is the code a line-by-line
transcription of anyone's source — it is written against this repository's own
`nn.Conv2D` / `nn.ModelBase` (`core/leras`), not against `torch.nn` directly.

**That architecture has more than one implementation, and they are not under
the same terms.** S3FD is a detector published in a 2017 paper; anyone may
implement it, and no list of implementations can be closed. The ones below are
**examples, not a complete list**:

* <https://github.com/sfzhang15/SFD>, from the authors of the S3FD paper,
  ships **no license file**: the license endpoint of the GitHub API for that
  repository, `api.github.com/repos/sfzhang15/SFD/license`, answered
  `Not Found` when it was queried on 2026-08-19. Its tree holds evaluation
  scripts and result files only — no model definition and no weight file, the
  trained Caffe model being offered through an off-site download link in its
  README — and the language GitHub reports for it is Matlab.
* <https://github.com/clcarwin/SFD_pytorch>, **MIT**,
  `Copyright (c) 2017 carwin`. A Python model carrying those same four layer
  names, which distributes **converted S3FD weights of its own**
  (`s3fd_convert.7z`, release `v0.1`).
* <https://github.com/1adrianb/face-alignment>, **BSD 3-Clause**, Adrian
  Bulat — the project named in the section above as the origin of `2DFAN.npy`
  and `3DFAN.npy`. It carries a Python S3FD model,
  `face_alignment/detection/sfd/net_s3fd.py`, with the same four layer names,
  and it distributes **S3FD weights of its own** (`s3fd-619a316812.pth`).

**This repository cannot say which license applies to `S3FD.npy`, and claims
none.** Each license text reproduced in this file is reproduced for the weight
files its own section names, and covers those. One of them happens to reach
further: had `S3FD.npy` come through face-alignment, the BSD 3-Clause text
printed above for `2DFAN.npy` and `3DFAN.npy` is the same text that would be
required for it — same copyright holder, same license. For an origin under any
other terms, this file does not reproduce the notice that would be required.
Which of these is the case is not known here.
