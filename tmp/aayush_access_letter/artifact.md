# Template execution contract

- Reference: `/Users/shahilshaik/Documents/Attention is Transformed/docs/cv/ACCESS_Advisor_Support_Letter_Draft.docx`
- SHA-256: `8b86a1b03aa18db9c43f1befe61195ef3794ec4b25046c241a37e274edee6e30`
- Evidence: `reference-render/page-1.png`, `reference-style-evidence.json`, and the section audit captured during this task.
- Page count: 1. Section count: 1.

## Page system

- US Letter portrait, 8.5 x 11 inches.
- Margins: left 1.0 in, right 1.0 in, top 1.2 in, bottom 0.85 in.
- One section, no first-page variation, empty header, dedicated footer.
- Preserve the one-page layout and footer position.

## Typography and paragraph rhythm

- Calibri 11 pt throughout the body.
- Normal paragraph style with direct paragraph spacing matching the source.
- Date paragraph: 0 pt before, 12 pt after.
- Recipient lines: 0 pt before/after; subject line is bold with 12 pt after.
- Salutation: 10 pt after.
- Body paragraphs: 8 pt after.
- Closing support paragraph: 14 pt after.
- `Sincerely,`: 34 pt after to reserve signature space.
- Signer name: bold; remaining signature-title lines regular with no added space.
- Footer: centered italic gray draft instruction; preserve its styling and location.

## Content flow and editable slots

- `word/document.xml`, body paragraph 1: replace date with August 14, 2026.
- Body paragraph 4: replace Shahil with Aayush in the subject line; preserve bold.
- Body paragraph 6: replace the student/request sentence with Aayush's status; keep the request phrase bold.
- Body paragraph 7: replace the title and project-description content; keep only the project title italic.
- Body paragraph 8: replace compute rationale while preserving paragraph formatting.
- Body paragraph 9: replace Shahil with Aayush and retain the guidance language.
- Body paragraph 10: replace the closing research summary and Shahil with Aayush.
- Body paragraphs 2-3, 5, and 11-15: preserve text and formatting.
- `word/footer1.xml`: preserve the draft-for-review instruction unchanged.

## Package preservation

Preserve all package parts and relationships other than the intended text nodes in
`word/document.xml`. In particular, preserve styles, stylesWithEffects, numbering,
settings, font table, theme, custom XML, footer, relationships, content types, and the
thumbnail. The reference contains no tables, images in the document body, comments,
tracked changes, fields, content controls, or non-empty header content.

## Fidelity gates

- The retained reference must remain byte-for-byte unchanged.
- The final must remain one page with identical page geometry and signature spacing.
- No unexplained change to the footer, styles, relationships, or recurring page furniture.
- No clipping, overlap, missing glyphs, orphan line, or unexpected page break.
- All new prose must identify Aayush and the robotics-centered GT-MHA project accurately.

