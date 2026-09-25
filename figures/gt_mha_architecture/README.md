# GT-MHA architecture figure

## Revised horizontal figure

`gt_mha_horizontal.svg` is the compact, integrated architecture illustration (1560 × 450, aspect ratio 3.47:1). At 6.75 inches wide it is approximately 1.95 inches tall before its caption. Open in Illustrator to edit; export as a vector PDF for the paper. `gt_mha_horizontal.png` is the 2× preview, `horizontal_caption.tex` is its caption, and `build_horizontal.py` regenerates the vector artwork. The original three-panel version below is retained for comparison.

The solid colored routes carry shared base features to the heads; dashed routes carry transformation parameters. The single surface represents the exact-map construction schematically for the two separately learned QK/value banks. The heatmaps illustrate potential patterns, not measured head diversity. Three base families and six heads are illustrative, not a claim about the experiment configuration.

- `gt_mha_architecture.svg`: editable vector artwork; open directly in Adobe Illustrator.
- `gt_mha_architecture.png`: high-resolution raster preview.
- `caption.tex`: suggested manuscript caption.
- `build_figure.py`: reproducible SVG source (Python standard library only).

The geometric inset is a vector reinterpretation of the supplied illustration, not an embedded raster. All paths, arrows, grids, boxes, and text remain editable. Named groups separate the attention data flow, transformation construction, and geometric inset. Fonts are Arial/Helvetica for labels and STIX Two Text/Times New Roman for mathematics, with fallbacks. Keep text editable in the working SVG; check substitutions when opening in Illustrator and outline a duplicate only for final delivery if needed.

For Overleaf, open the SVG in Illustrator and save a PDF using the artboard bounds, preserving vectors and embedding fonts. Use the PDF at full text width; the three-panel composition is intended as a full-width architecture figure. The PNG is a preview, not the preferred publication asset.

Conventions: the attention mask represents causal or padding masking as appropriate. Value transformation after aggregation is the algebraically equivalent presentation. Projection biases and optional implementation-specific scaling are omitted for clarity. The generator construction depicts softmax mixing. The inset is illustrative: the group is not literally a two-dimensional surface, the depicted generator span need not fill the Lie algebra, and the individual head transformations need not lie on a common curve. No claim of surjectivity of the exponential map is intended. The point I identifies the tangent-space base point; the tangent-vector origin corresponds to zero under its identification with the Lie algebra.

The value-path symbols in the caption are locally defined for clarity; align them with final manuscript notation before submission. No changes were made to Overleaf.
