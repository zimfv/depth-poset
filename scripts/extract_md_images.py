#!/usr/bin/env python3

import base64
import hashlib
import re
import sys
from pathlib import Path


# Matches:
# ![caption](data:image/png;base64,...)
# ![](data:image/jpeg;base64,...)
PATTERN = re.compile(
    r'<img\b([^>]*?)src=["\']'
    r'data:image/([a-zA-Z0-9.+-]+);base64,'
    r'([A-Za-z0-9+/=\s]+)'
    r'["\']([^>]*)>',
    re.IGNORECASE | re.MULTILINE,
)

EXTENSIONS = {
    "jpeg": "jpg",
    "jpg": "jpg",
    "png": "png",
    "gif": "gif",
    "webp": "webp",
    "svg+xml": "svg",
}


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} FILE.md")
        sys.exit(1)

    md_path = Path(sys.argv[1])
    text = md_path.read_text(encoding="utf-8")

    # e.g. reports/transpositions_stats-abstract.assets/
    assets_dir = md_path.parent / f"{md_path.stem}.assets"
    assets_dir.mkdir(exist_ok=True)

    count = 0


    def replace(match):
        nonlocal count

        before = match.group(1)
        image_type = match.group(2).lower()
        encoded = re.sub(r"\s+", "", match.group(3))
        after = match.group(4)

        data = base64.b64decode(encoded)

        ext = EXTENSIONS.get(image_type, image_type)

        digest = hashlib.sha256(data).hexdigest()[:12]
        filename = f"image-{digest}.{ext}"

        output = assets_dir / filename
        if not output.exists():
            output.write_bytes(data)

        count += 1

        relative = f"{assets_dir.name}/{filename}"

        # Сохраняем сам <img>, включая остальные атрибуты.
        return f'<img{before}src="{relative}"{after}>'


    new_text = PATTERN.sub(replace, text)

    # Keep the original untouched.
    output_md = md_path.with_name(md_path.stem + "-github.md")
    output_md.write_text(new_text, encoding="utf-8")

    print(f"Extracted/referenced {count} images")
    print(f"Markdown: {output_md}")
    print(f"Images:   {assets_dir}")


if __name__ == "__main__":
    main()