import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("TOC_DATA", ROOT / "data"))

IMMICH_CONTAINER = os.environ.get("TOC_CONTAINER", "immich_postgres")
IMMICH_DB = os.environ.get("TOC_DB", "immich")

LIBRARY_DIR = Path("/mnt/hdd/Library")
THUMBS_DIR = LIBRARY_DIR / "thumbs"

MODEL_ID = "google/siglip2-base-patch16-224"

ISO = "'YYYY-MM-DD\"T\"HH24:MI:SS.US\"Z\"'"

SQL = f"""
SELECT
  a.id,
  a."ownerId",
  a.type,
  a.width,
  a.height,
  a."originalFileName",
  a."originalPath",
  encode(a.checksum, 'hex')                        AS checksum,
  encode(a.thumbhash, 'hex')                       AS thumbhash,
  to_char(a."fileCreatedAt" AT TIME ZONE 'UTC', {ISO})  AS file_created_at,
  to_char(a."fileModifiedAt" AT TIME ZONE 'UTC', {ISO}) AS file_modified_at,
  to_char(a."localDateTime" AT TIME ZONE 'UTC', {ISO})  AS local_date_time,
  to_char(a."createdAt" AT TIME ZONE 'UTC', {ISO})      AS imported_at,
  to_char(a."deletedAt" AT TIME ZONE 'UTC', {ISO})      AS deleted_at,
  a."isFavorite"                                   AS is_favorite,
  a.visibility,
  a."isEdited"                                     AS is_edited,
  a."stackId"                                      AS stack_id,
  a."duplicateId"                                  AS duplicate_id,
  e.make,
  e.model,
  e.iso,
  e."fNumber"                                      AS f_number,
  e."focalLength"                                  AS focal_length,
  e."fileSizeInByte"                               AS file_size,
  e.latitude,
  e.longitude,
  e.city,
  e.country,
  e."exifImageWidth"                               AS exif_width,
  e."exifImageHeight"                              AS exif_height,
  to_char(e."dateTimeOriginal" AT TIME ZONE 'UTC', {ISO}) AS exif_taken_at,
  (SELECT count(*) FROM asset_ocr o WHERE o."assetId" = a.id)                      AS ocr_rows,
  (SELECT coalesce(sum(length(o.text)), 0) FROM asset_ocr o WHERE o."assetId" = a.id) AS ocr_chars,
  (SELECT count(*) FROM asset_face f WHERE f."assetId" = a.id AND f."deletedAt" IS NULL) AS face_count
FROM asset a
LEFT JOIN asset_exif e ON e."assetId" = a.id
WHERE a.type = 'IMAGE'
ORDER BY a.id
"""
