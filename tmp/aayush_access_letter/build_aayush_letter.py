from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from xml.sax.saxutils import escape


ROOT = Path("/Users/shahilshaik/Documents/Attention is Transformed")
SOURCE = ROOT / "docs/cv/ACCESS_Advisor_Support_Letter_Draft.docx"
OUTPUT = ROOT / "docs/cv/Aayush_Rai_ACCESS_Advisor_Support_Letter_for_Signature.docx"


def xml_text(value: str) -> str:
    return escape(value, {'"': "&quot;"})


replacements = {
    "July 27, 2026": "August 14, 2026",
    "Re: Advisor Support for Shahil Shaik’s Explore ACCESS Request":
        "Re: Advisor Support for Aayush Rai’s Explore ACCESS Request",
    "I am writing in support of ": "I am writing in support of ",
    "Shahil Shaik’s Explore ACCESS allocation request":
        "Aayush Rai’s Explore ACCESS allocation request",
    " for NCSA DeltaAI. Shahil is a Ph.D. Candidate and Graduate Research Assistant in the Department of Mechanical Engineering at Clemson University, and I serve as his faculty advisor.":
        " for NCSA DeltaAI. Aayush is a Ph.D. student in the Department of Mechanical Engineering at Clemson University, and I serve as his faculty advisor.",
    "“Evaluating Lie-Generated Metric Attention for Efficient Language and Vision-Language-Action Models.”":
        "“Geometry-Structured Attention for Robust Vision-Language Control.”",
    " The research will evaluate Lie-Generated Metric Attention as a parameter- and memory-efficient alternative to conventional multi-head attention. The planned work includes controlled language-model experiments using TinyStories and, where appropriate, a WikiHow corpus, together with vision-language-action experiments on the LIBERO robot-learning benchmark.":
        " The research will evaluate geometry-structured multi-head attention (GT-MHA), implemented through Lie-generated metric attention, as a parameter- and memory-efficient alternative to conventional multi-head attention. The planned work centers on vision-language-action policies using the LIBERO robot-learning benchmark, with controls-oriented evaluation of task success, convergence, and robustness to changes in viewpoint, scene configuration, object arrangement, and task instruction.",
    "NCSA DeltaAI is appropriate for this work because the experiments require H100/GH200-class GPU memory and mixed-precision throughput for repeated transformer training, multimodal vision processing, and selected multi-GPU scaling studies. The project will compare model quality, convergence, parameter efficiency, key-value cache requirements, throughput, peak GPU memory, and task performance using reproducible experimental configurations.":
        "NCSA DeltaAI is appropriate for this work because the experiments require H100/GH200-class GPU memory and mixed-precision throughput for repeated multimodal policy training, vision processing, robustness studies, and selected multi-GPU verification. The project will compare task success, generalization, convergence, parameter efficiency, key-value cache requirements, throughput, and peak GPU memory using reproducible experimental configurations.",
    "I intend to engage with and guide the computational and research activities conducted under this allocation. My guidance will include oversight of experimental design, responsible use of the requested resources, interpretation of results, reproducibility, and dissemination of the resulting research. I will remain involved throughout the allocation period and will support Shahil in meeting applicable ACCESS reporting and resource-use requirements.":
        "I intend to engage with and guide the computational and research activities conducted under this allocation. My guidance will include oversight of experimental design, responsible use of the requested resources, interpretation of results, reproducibility, and dissemination of the resulting research. I will remain involved throughout the allocation period and will support Aayush in meeting applicable ACCESS reporting and resource-use requirements.",
    "I believe the requested resources will materially advance this research and provide an appropriate computational environment for rigorous evaluation of the proposed attention and robot-learning methods. I support Shahil’s request for ACCESS resources.":
        "I believe the requested resources will materially advance Aayush’s dissertation research and provide an appropriate computational environment for rigorous evaluation of the proposed attention and robot-learning methods. I support Aayush’s request for ACCESS resources.",
}


with ZipFile(SOURCE, "r") as src:
    entries = {info.filename: (info, src.read(info.filename)) for info in src.infolist()}

document_xml = entries["word/document.xml"][1].decode("utf-8")
for old, new in replacements.items():
    old_xml = xml_text(old)
    if document_xml.count(old_xml) != 1:
        raise RuntimeError(f"Expected exactly one match for: {old!r}")
    document_xml = document_xml.replace(old_xml, xml_text(new), 1)

entries["word/document.xml"] = (
    entries["word/document.xml"][0],
    document_xml.encode("utf-8"),
)

with ZipFile(OUTPUT, "w", compression=ZIP_DEFLATED) as dst:
    for name, (info, data) in entries.items():
        dst.writestr(info, data)

print(OUTPUT)

