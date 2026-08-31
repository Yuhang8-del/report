# Auxiliary datasets: local-only import policy

Auxiliary sources are indexed only from caller-provided local directories. The importer does not download data, copy images, modify source images, infer boxes, or invent labels. Before acquiring any data, recheck the source page and its current terms; this repository does not treat this document as permission to download or redistribute a dataset.

## Fruits-360

Fruits-360 is recorded with license **CC BY-SA 4.0**. Recheck the source page before downloading a particular release. Run the local importer with an explicit source version, source-page URL, license facts, input directory, and output directory:

```powershell
conda run -n fruit-ssod python -m fruit_ssod.cli.import_auxiliary_data fruits360 `
  --images-root D:\data\fruits360\Training `
  --output-dir D:\data\manifests\fruits360 `
  --source-version <release> --source-page <checked-source-page-url> `
  --license-name "CC BY-SA 4.0" --license-url <checked-license-url>
```

The input is expected to have category directories below `--images-root`. Approved directory names are resolved only for curation validation; the original directory/category identifier remains in the manifest. Every usable image becomes an `UnlabeledImageRecord` with `split=train_pool` and `label_status=unlabeled`. There are no detection boxes or object class labels. Unknown categories and unreadable images are retained as actionable manifest rejections.

## FruitDet

FruitDet is recorded with license **CC BY 4.0**. Recheck the source page before downloading a particular release. It must be supplied as a local COCO-style JSON object with `images`, `categories`, and `annotations`, plus a caller-provided image root used to contain every relative `file_name`.

```powershell
conda run -n fruit-ssod python -m fruit_ssod.cli.import_auxiliary_data fruitdet `
  --annotations D:\data\fruitdet\annotations.json `
  --images-root D:\data\fruitdet\images `
  --output-dir D:\data\manifests\fruitdet `
  --source-version <release> --source-page <checked-source-page-url> `
  --license-name "CC BY 4.0" --license-url <checked-license-url>
```

FruitDet is always `split=external_test`, with human/labeled annotations. The CLI refuses `train_pool`, `validation`, and `test` values, so its records cannot be merged into the primary dataset partitions. Its categories are mapped through `limited_external_set`: Apple, Banana, Orange, and Strawberry are accepted; Pineapple is absent and is recorded as an unsupported-category rejection rather than remapped. COCO IDs, dimensions, duplicate annotation IDs, relative-path containment, category mappings, and finite in-bounds XYWH geometry are validated before annotations are emitted.

Each command creates only the explicit `--output-dir` and writes `manifest.json` there. The manifest records source name/version/page/license, original source identifiers, split/status, accepted-record count, and rejection count. No credentials or personal paths are stored in project configuration.

## Kaggle Fruit Detection recovery candidate

The public `lakshaytyagi01/fruit-detection` release is recorded as CC0 and
contains YOLO-format boxes for Apple, Banana, Orange and Pineapple. The
reviewed version-1 archive is locally staged with SHA-256
`b759435fc06e34cf900129dc3144535f38f1c6233247fe86faf862e1177545c5`, but it is
not part of the active v10 experiment. Safely extract that reviewed archive
under the configured raw-data root, then index it locally:

```powershell
conda run -n fruit-ssod python -m fruit_ssod.cli.import_auxiliary_data kaggle-fruit-detection `
  --dataset-root E:\fruit_ssod_runtime\data\fruit_ssod\raw\kaggle_fruit_detection `
  --data-yaml E:\fruit_ssod_runtime\data\fruit_ssod\raw\kaggle_fruit_detection\data.yaml `
  --output-dir E:\fruit_ssod_runtime\data\fruit_ssod\interim\kaggle_fruit_detection\v1 `
  --source-version 1 --source-page https://www.kaggle.com/datasets/lakshaytyagi01/fruit-detection `
  --license-name "CC0: Public Domain"
```

The importer makes no network request and accepts only safe
`images/<partition>`/`labels/<partition>` pairs with in-bounds YOLO boxes. It
maps only the five fixed canonical fruit classes; all other source categories
are preserved as rejections. The result is always `train_pool` data and must
be merged, cleaned, split and audited as a new protocol before training. It
does not replace the separate FruitDet external-test policy.

## Zenodo Strawberry detection recovery

Zenodo record `6126677` is a locally staged CC BY 4.0 source with
YOLO boxes for `ripe`, `unripe`, and `peduncle`. Only the first two source
classes are explicit Strawberry aliases. Peduncles are retained as manifest
rejections, never remapped to fruit. Its original `training` and `validation`
directories are discarded in favour of a fresh project-level split. Its
verified archive SHA-256 is
`d4664ccd6288e077934e175c281f6a8146da5bf194a1be0e96bca8336faf320c`; it is not
part of v10 and may only enter a new audited recovery protocol.

```powershell
conda run -n fruit-ssod python -m fruit_ssod.cli.import_auxiliary_data zenodo-strawberry `
  --dataset-root E:\fruit_ssod_runtime\data\fruit_ssod\raw\zenodo_strawberry_6126677\extracted_sanitized\strawberries `
  --data-yaml E:\fruit_ssod_runtime\data\fruit_ssod\raw\zenodo_strawberry_6126677\extracted_sanitized\strawberries\strawberries.yaml `
  --output-dir E:\fruit_ssod_runtime\data\fruit_ssod\interim\zenodo_strawberry_6126677\v1 `
  --source-version 6126677 --source-page https://zenodo.org/records/6126677 `
  --license-name "CC BY 4.0" --license-url https://creativecommons.org/licenses/by/4.0/
```

## Berremangra Orange

The Berremangra Orange dataset (Zenodo record `20481328`, v1, CC BY 4.0) is
an independently downloaded supplementary source. Its labels are YOLO polygon
segments for one Orange class; the importer converts each polygon to its
enclosing detection rectangle while recording the conversion in the manifest.
The original source partition is not reused: imported records are `train_pool`
and must receive a new project-level cleaning, duplicate and split audit.

```powershell
conda run -n fruit-ssod python -m fruit_ssod.cli.import_auxiliary_data berremangra-orange `
  --dataset-root "E:\fruit_ssod_runtime\data\fruit_ssod\raw\berremangra_orange\extracted_sanitized\Berremangra Orange" `
  --output-dir E:\fruit_ssod_runtime\data\fruit_ssod\interim\berremangra_orange\v1 `
  --source-version v1 --source-page https://zenodo.org/records/20481328 `
  --license-name "CC BY 4.0" --license-url https://creativecommons.org/licenses/by/4.0/
```

## deepNIR fruit detection

The supplementary deepNIR release is Zenodo record `6324489` (CC BY 4.0).
Its archive contains one YOLO directory per source fruit, but several
`data.yaml` files use the placeholder class name `1` (and one has malformed
YAML). The importer therefore never uses those fields for class mapping. It
imports only the reviewed `apple`, `orange` and `strawberry` directories,
requires their single-class label ID to be `0`, and records the directory name
as the auditable source category. The source's original train/valid split is
discarded so the project can create a leakage-controlled fresh split.

```powershell
conda run -n fruit-ssod python -m fruit_ssod.cli.import_auxiliary_data deepnir `
  --dataset-root E:\fruit_ssod_runtime\data\fruit_ssod\raw\deepnir\extracted_sanitized `
  --output-dir E:\fruit_ssod_runtime\data\fruit_ssod\interim\deepnir\record-6324489 `
  --source-version record-6324489-2022-03-15 `
  --source-page https://zenodo.org/records/6324489 `
  --license-name "CC BY 4.0" --license-url https://creativecommons.org/licenses/by/4.0/
```

The manifest contains only Apple, Orange and Strawberry records. It must be
merged, cleaned, split and audited as a new project protocol before it can be
used for training; it cannot supply the missing Banana or Pineapple class by
inference.

## Hugging Face crop/plant recovery candidate

The reviewed `devshaheen/100_crops_plants_object_detection_25k_image_dataset`
archive supplies YOLOv5 boxes for 100 crop/plant categories. Its Hugging Face
card declares MIT, while the embedded upstream Roboflow `data.yaml` declares
CC BY 4.0. The importer therefore records **CC BY 4.0**, the more restrictive
of the two stated terms, and retains both source pages in the curation log.
Only the explicitly reviewed source labels `apple fruit`, `banana`, `orange`,
`strawberry` and `pineapple` map to the five fixed detector classes; all other
classes become auditable rejections. The source's original partitions are not
reused as project evaluation partitions.

```powershell
conda run -n fruit-ssod python -m fruit_ssod.cli.import_auxiliary_data hf-crop-plant `
  --dataset-root 'E:\fruit_ssod_runtime\data\fruit_ssod\raw\hf_crop_plant_25k\extracted_sanitized_tar\leaflogic object detection.v5i.yolov5pytorch' `
  --data-yaml 'E:\fruit_ssod_runtime\data\fruit_ssod\raw\hf_crop_plant_25k\extracted_sanitized_tar\leaflogic object detection.v5i.yolov5pytorch\data.yaml' `
  --output-dir E:\fruit_ssod_runtime\data\fruit_ssod\interim\hf_crop_plant_25k\v5 `
  --source-version v5 --source-page https://huggingface.co/datasets/devshaheen/100_crops_plants_object_detection_25k_image_dataset `
  --license-name 'CC BY 4.0' --license-url https://creativecommons.org/licenses/by/4.0/
```

The output is a fresh labeled train-pool source. It must be merged with the
primary source, cleaned, split and audited under a new protocol before any
training; it cannot be used as a replacement test set or to claim acceptance.
