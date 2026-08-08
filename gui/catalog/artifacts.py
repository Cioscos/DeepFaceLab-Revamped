"""The workflow's artifact table: files and directories that steps consume, produce or modify.

Transcribed field by field from the `[[artefatto]]` entries of the workflow
formalization. `origin` keeps the literal Italian values used there
("prodotto", "fornito-dall-utente", "asset") so the sync guard can compare
without translation.
"""
from gui.catalog.model import ArtifactDef

ARTIFACTS = (
    ArtifactDef(
        name="video_src",
        patterns=("{WORKSPACE}/data_src.*",),
        origin="fornito-dall-utente",
    ),
    ArtifactDef(
        name="video_dst",
        patterns=("{WORKSPACE}/data_dst.*",),
        origin="fornito-dall-utente",
    ),
    ArtifactDef(
        name="frame_src",
        patterns=("{WORKSPACE}/data_src",),
        origin="prodotto",
    ),
    ArtifactDef(
        name="frame_dst",
        patterns=("{WORKSPACE}/data_dst",),
        origin="prodotto",
    ),
    ArtifactDef(
        name="faceset_src",
        patterns=("{WORKSPACE}/data_src/aligned",),
        origin="prodotto",
    ),
    ArtifactDef(
        name="faceset_dst",
        patterns=("{WORKSPACE}/data_dst/aligned",),
        origin="prodotto",
    ),
    ArtifactDef(
        name="debug_dst",
        patterns=("{WORKSPACE}/data_dst/aligned_debug",),
        origin="prodotto",
    ),
    ArtifactDef(
        name="modello",
        patterns=("{WORKSPACE}/model",),
        origin="prodotto",
    ),
    ArtifactDef(
        name="merged",
        patterns=("{WORKSPACE}/data_dst/merged",),
        origin="prodotto",
    ),
    ArtifactDef(
        name="merged_mask",
        patterns=("{WORKSPACE}/data_dst/merged_mask",),
        origin="prodotto",
    ),
    ArtifactDef(
        name="risultato",
        patterns=("{WORKSPACE}/result.*",),
        origin="prodotto",
    ),
    ArtifactDef(
        name="risultato_mask",
        patterns=("{WORKSPACE}/result_mask.*",),
        origin="prodotto",
    ),
    ArtifactDef(
        name="xseg_generico",
        patterns=("{INTERNAL}/model_generic_xseg",),
        origin="asset",
    ),
    ArtifactDef(
        name="pretrain",
        patterns=("{INTERNAL}/pretrain_faces",),
        origin="asset",
    ),
)
