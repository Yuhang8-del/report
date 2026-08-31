"use strict";

/**
 * Build a page-faithful DOCX companion from rendered PDF page images.
 *
 * The output intentionally prioritizes reliable visual round-tripping over
 * paragraph-level editing. The fully editable report remains final_report.docx.
 */

const fs = require("fs");
const path = require("path");
const {
  AlignmentType,
  Document,
  HorizontalPositionRelativeFrom,
  ImageRun,
  Packer,
  PageOrientation,
  Paragraph,
  SectionType,
  TextWrappingType,
  VerticalPositionRelativeFrom,
} = require("docx");

const LETTER_WIDTH_TWIPS = 12240;
const LETTER_HEIGHT_TWIPS = 15840;
// docx converts these display pixels to EMU using 96 dpi: 816 x 1056 = 8.5 x 11 in.
const LETTER_WIDTH_PX = 816;
const LETTER_HEIGHT_PX = 1056;

function naturalSort(a, b) {
  return a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" });
}

async function main() {
  const pageDir = process.argv[2];
  const outputPath = process.argv[3];
  if (!pageDir || !outputPath) {
    throw new Error("Usage: node build_high_fidelity_word.js <page_png_dir> <output.docx>");
  }

  const pages = fs
    .readdirSync(pageDir)
    .filter((name) => /^page-\d+\.png$/i.test(name))
    .sort(naturalSort);
  if (pages.length === 0) {
    throw new Error(`No page-*.png images found in ${pageDir}`);
  }

  const sections = pages.map((name, index) => ({
    properties: {
      type: index === 0 ? undefined : SectionType.NEXT_PAGE,
      page: {
        size: {
          width: LETTER_WIDTH_TWIPS,
          height: LETTER_HEIGHT_TWIPS,
          orientation: PageOrientation.PORTRAIT,
        },
        margin: {
          top: 0,
          right: 0,
          bottom: 0,
          left: 0,
          header: 0,
          footer: 0,
          gutter: 0,
        },
      },
    },
    children: [
      new Paragraph({
        alignment: AlignmentType.LEFT,
        spacing: { before: 0, after: 0, line: 1, lineRule: "exact" },
        children: [
          new ImageRun({
            type: "png",
            data: fs.readFileSync(path.join(pageDir, name)),
            transformation: {
              width: LETTER_WIDTH_PX,
              height: LETTER_HEIGHT_PX,
            },
            floating: {
              horizontalPosition: {
                relative: HorizontalPositionRelativeFrom.PAGE,
                offset: 0,
              },
              verticalPosition: {
                relative: VerticalPositionRelativeFrom.PAGE,
                offset: 0,
              },
              wrap: { type: TextWrappingType.NONE },
              behindDocument: false,
              allowOverlap: true,
              lockAnchor: true,
            },
            altText: {
              title: `Final report page ${index + 1}`,
              description: `High-fidelity rendering of final_report.pdf page ${index + 1}`,
              name: `page-${index + 1}`,
            },
          }),
        ],
      }),
    ],
  }));

  const doc = new Document({
    creator: "Fruit SSOD Project",
    title: "Semi-Supervised Fruit Detection and Novel Category Discovery",
    description: "High-fidelity Word companion of the final PDF report",
    styles: {
      default: {
        document: {
          run: { font: "Arial", size: 2 },
          paragraph: { spacing: { before: 0, after: 0 } },
        },
      },
    },
    sections,
  });

  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, await Packer.toBuffer(doc));
  process.stdout.write(`${outputPath}\n${pages.length} pages\n`);
}

main().catch((error) => {
  process.stderr.write(`${error.stack || error.message}\n`);
  process.exitCode = 1;
});
