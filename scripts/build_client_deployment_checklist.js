"use strict";

const fs = require("fs");
const {
  AlignmentType,
  BorderStyle,
  Document,
  Footer,
  Header,
  HeadingLevel,
  LevelFormat,
  PageNumber,
  Packer,
  Paragraph,
  ShadingType,
  Table,
  TableCell,
  TableRow,
  TextRun,
  VerticalAlign,
  WidthType,
} = require("docx");

const input = process.argv[2];
const output = process.argv[3];
if (!input || !output) {
  throw new Error("Usage: node build_client_deployment_checklist.js <input.md> <output.docx>");
}

const BODY_FONT = "Microsoft YaHei";
const CODE_FONT = "Consolas";
const PAGE_WIDTH = 11906;
const PAGE_HEIGHT = 16838;
const MARGIN = 1134;
const CONTENT_WIDTH = PAGE_WIDTH - MARGIN * 2;
const border = { style: BorderStyle.SINGLE, size: 4, color: "B8BEC6" };
const borders = { top: border, bottom: border, left: border, right: border };

function inlineRuns(text, options = {}) {
  const runs = [];
  const pattern = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  let cursor = 0;
  for (const match of text.matchAll(pattern)) {
    if (match.index > cursor) {
      runs.push(new TextRun({ text: text.slice(cursor, match.index), ...options }));
    }
    const token = match[0];
    if (token.startsWith("**")) {
      runs.push(new TextRun({ text: token.slice(2, -2), bold: true, ...options }));
    } else {
      runs.push(new TextRun({
        text: token.slice(1, -1),
        font: CODE_FONT,
        color: "27364A",
        size: options.size || 19,
      }));
    }
    cursor = match.index + token.length;
  }
  if (cursor < text.length) {
    runs.push(new TextRun({ text: text.slice(cursor), ...options }));
  }
  return runs.length ? runs : [new TextRun({ text: " ", ...options })];
}

function bodyParagraph(text, extra = {}) {
  return new Paragraph({
    spacing: { after: 100, line: 300 },
    ...extra,
    children: inlineRuns(text, { font: BODY_FONT, size: 21, color: "20252B" }),
  });
}

function heading(text, level) {
  const map = {
    1: HeadingLevel.TITLE,
    2: HeadingLevel.HEADING_1,
    3: HeadingLevel.HEADING_2,
  };
  return new Paragraph({
    heading: map[level] || HeadingLevel.HEADING_3,
    keepNext: true,
    children: [new TextRun({ text, font: BODY_FONT })],
  });
}

function codeBlock(lines) {
  return new Table({
    width: { size: CONTENT_WIDTH, type: WidthType.DXA },
    columnWidths: [CONTENT_WIDTH],
    margins: { top: 130, bottom: 130, left: 180, right: 180 },
    rows: [
      new TableRow({
        children: [
          new TableCell({
            width: { size: CONTENT_WIDTH, type: WidthType.DXA },
            borders,
            shading: { fill: "F3F5F7", type: ShadingType.CLEAR },
            children: lines.map((line) => new Paragraph({
              spacing: { after: 20, line: 240 },
              children: [new TextRun({ text: line || " ", font: CODE_FONT, size: 18, color: "1D2733" })],
            })),
          }),
        ],
      }),
    ],
  });
}

function tableWidths(columnCount) {
  if (columnCount === 2) return [Math.round(CONTENT_WIDTH * 0.34), CONTENT_WIDTH - Math.round(CONTENT_WIDTH * 0.34)];
  if (columnCount === 3) return [2500, 3900, CONTENT_WIDTH - 6400];
  const base = Math.floor(CONTENT_WIDTH / columnCount);
  return Array.from({ length: columnCount }, (_, i) => i === columnCount - 1 ? CONTENT_WIDTH - base * (columnCount - 1) : base);
}

function markdownTable(rows) {
  const widths = tableWidths(rows[0].length);
  return new Table({
    width: { size: CONTENT_WIDTH, type: WidthType.DXA },
    columnWidths: widths,
    margins: { top: 90, bottom: 90, left: 100, right: 100 },
    rows: rows.map((row, rowIndex) => new TableRow({
      tableHeader: rowIndex === 0,
      cantSplit: true,
      children: row.map((value, columnIndex) => new TableCell({
        width: { size: widths[columnIndex], type: WidthType.DXA },
        borders,
        verticalAlign: VerticalAlign.CENTER,
        shading: { fill: rowIndex === 0 ? "E9EDF2" : "FFFFFF", type: ShadingType.CLEAR },
        children: [new Paragraph({
          spacing: { after: 30, line: 250 },
          children: inlineRuns(value || " ", {
            font: BODY_FONT,
            size: rowIndex === 0 ? 19 : 18,
            bold: rowIndex === 0,
            color: "20252B",
          }),
        })],
      })),
    })),
  });
}

function callout(text) {
  return new Table({
    width: { size: CONTENT_WIDTH, type: WidthType.DXA },
    columnWidths: [CONTENT_WIDTH],
    margins: { top: 130, bottom: 130, left: 180, right: 180 },
    rows: [new TableRow({ children: [new TableCell({
      width: { size: CONTENT_WIDTH, type: WidthType.DXA },
      borders: {
        top: { style: BorderStyle.SINGLE, size: 4, color: "9AA7B5" },
        bottom: { style: BorderStyle.SINGLE, size: 4, color: "9AA7B5" },
        left: { style: BorderStyle.SINGLE, size: 18, color: "3E617F" },
        right: { style: BorderStyle.SINGLE, size: 4, color: "9AA7B5" },
      },
      shading: { fill: "F4F6F8", type: ShadingType.CLEAR },
      children: [bodyParagraph(text, { spacing: { after: 0, line: 280 } })],
    })] })],
  });
}

function parseMarkdown(markdown) {
  const lines = markdown.replace(/\r/g, "").split("\n");
  const children = [];
  let i = 0;
  let numberedGroup = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { i += 1; continue; }

    if (line.startsWith("```")) {
      const block = [];
      i += 1;
      while (i < lines.length && !lines[i].startsWith("```")) {
        block.push(lines[i]);
        i += 1;
      }
      i += 1;
      children.push(codeBlock(block));
      children.push(new Paragraph({ spacing: { after: 80 }, children: [] }));
      continue;
    }

    const h = /^(#{1,4})\s+(.+)$/.exec(line);
    if (h) {
      children.push(heading(h[2], h[1].length));
      i += 1;
      continue;
    }

    if (line.startsWith("> ")) {
      children.push(callout(line.slice(2)));
      children.push(new Paragraph({ spacing: { after: 80 }, children: [] }));
      i += 1;
      continue;
    }

    if (line.trim().startsWith("|") && i + 1 < lines.length && /^\s*\|?[\s:|-]+\|\s*$/.test(lines[i + 1])) {
      const rows = [];
      const splitRow = (value) => value.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((cell) => cell.trim());
      rows.push(splitRow(line));
      i += 2;
      while (i < lines.length && lines[i].trim().startsWith("|")) {
        rows.push(splitRow(lines[i]));
        i += 1;
      }
      children.push(markdownTable(rows));
      children.push(new Paragraph({ spacing: { after: 100 }, children: [] }));
      continue;
    }

    if (/^\d+\.\s+/.test(line)) {
      numberedGroup += 1;
      const reference = `numbered-${numberedGroup}`;
      while (i < lines.length && /^\d+\.\s+/.test(lines[i])) {
        const text = lines[i].replace(/^\d+\.\s+/, "");
        children.push(new Paragraph({
          numbering: { reference, level: 0 },
          spacing: { after: 70, line: 280 },
          children: inlineRuns(text, { font: BODY_FONT, size: 21, color: "20252B" }),
        }));
        i += 1;
      }
      continue;
    }

    children.push(bodyParagraph(line));
    i += 1;
  }
  return { children, numberedGroups: numberedGroup };
}

async function main() {
  const parsed = parseMarkdown(fs.readFileSync(input, "utf8"));
  const numbering = [];
  for (let i = 1; i <= parsed.numberedGroups; i += 1) {
    numbering.push({
      reference: `numbered-${i}`,
      levels: [{
        level: 0,
        format: LevelFormat.DECIMAL,
        text: "%1.",
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 540, hanging: 300 } } },
      }],
    });
  }

  const doc = new Document({
    creator: "Fruit SSOD Project",
    title: "半监督水果检测项目——客户本地部署与使用清单",
    description: "Windows 本地部署、使用和复现验收清单",
    numbering: { config: numbering },
    styles: {
      default: { document: { run: { font: BODY_FONT, size: 21, color: "20252B" } } },
      paragraphStyles: [
        { id: "Title", name: "Title", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { font: BODY_FONT, size: 40, bold: true, color: "183A57" },
          paragraph: { alignment: AlignmentType.CENTER, spacing: { before: 240, after: 260 }, outlineLevel: 0 } },
        { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { font: BODY_FONT, size: 29, bold: true, color: "183A57" },
          paragraph: { spacing: { before: 260, after: 130 }, outlineLevel: 0, keepNext: true } },
        { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { font: BODY_FONT, size: 24, bold: true, color: "2B536F" },
          paragraph: { spacing: { before: 190, after: 100 }, outlineLevel: 1, keepNext: true } },
        { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { font: BODY_FONT, size: 22, bold: true, color: "34495E" },
          paragraph: { spacing: { before: 150, after: 80 }, outlineLevel: 2, keepNext: true } },
      ],
    },
    sections: [{
      properties: {
        page: {
          size: { width: PAGE_WIDTH, height: PAGE_HEIGHT },
          margin: { top: 980, right: MARGIN, bottom: 1050, left: MARGIN, header: 520, footer: 520 },
          pageNumbers: { start: 1, formatType: "decimal" },
        },
      },
      headers: { default: new Header({ children: [new Paragraph({
        alignment: AlignmentType.RIGHT,
        spacing: { after: 0 },
        children: [new TextRun({ text: "Fruit SSOD · Client Deployment Checklist", font: "Arial", size: 16, color: "68737E" })],
      })] }) },
      footers: { default: new Footer({ children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [
          new TextRun({ text: "Page ", font: "Arial", size: 16, color: "68737E" }),
          new TextRun({ children: [PageNumber.CURRENT], font: "Arial", size: 16, color: "68737E" }),
          new TextRun({ text: " / ", font: "Arial", size: 16, color: "68737E" }),
          new TextRun({ children: [PageNumber.TOTAL_PAGES], font: "Arial", size: 16, color: "68737E" }),
        ],
      })] }) },
      children: parsed.children,
    }],
  });

  fs.writeFileSync(output, await Packer.toBuffer(doc));
  process.stdout.write(`${output}\n`);
}

main().catch((error) => {
  process.stderr.write(`${error.stack || error.message}\n`);
  process.exitCode = 1;
});
