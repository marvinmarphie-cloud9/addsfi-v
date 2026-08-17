from io import BytesIO
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import (
    ParagraphStyle,
    getSampleStyleSheet,
)
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def _percentage(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def _value(
    payload: dict[str, Any],
    key: str,
    default: Any = "N/A",
) -> Any:
    value = payload.get(key)

    if value is None or value == "":
        return default

    return value


def _video_size(payload: dict[str, Any]) -> str:
    size_bytes = payload.get("size_bytes")

    if not size_bytes:
        return "N/A"

    try:
        size_mb = float(size_bytes) / (1024 * 1024)
        return f"{size_mb:.2f} MB"
    except (TypeError, ValueError):
        return "N/A"


def generate_prediction_report(
    payload: dict[str, Any],
) -> bytes:
    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="ADDFS Deepfake Analysis Report",
        author="ADDFS",
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        leading=24,
        spaceAfter=6,
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=10,
        textColor=colors.HexColor("#555555"),
        spaceAfter=18,
    )

    section_style = ParagraphStyle(
        "SectionTitle",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        spaceBefore=10,
        spaceAfter=8,
    )

    normal_style = ParagraphStyle(
        "NormalReport",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=14,
    )

    prediction = _value(
        payload,
        "prediction",
    )

    if prediction == "1_fake":
        prediction_label = "LIKELY DEEPFAKE"

        interpretation = (
            "The analyzed video was classified as likely "
            "manipulated by the ADDFS detection model."
        )
    else:
        prediction_label = "LIKELY AUTHENTIC"

        interpretation = (
            "The analyzed video was classified as likely "
            "authentic by the ADDFS detection model."
        )

    checkpoint = _value(
        payload,
        "model_checkpoint",
        _value(
            payload,
            "checkpoint",
        ),
    )

    if checkpoint != "N/A":
        checkpoint = Path(
            str(checkpoint)
        ).name

    story = []

    story.append(
        Paragraph(
            "ADDFS",
            title_style,
        )
    )

    story.append(
        Paragraph(
            "Automated Deepfake Detection System",
            subtitle_style,
        )
    )

    story.append(
        Paragraph(
            "Video Analysis Report",
            section_style,
        )
    )

    summary_data = [
        [
            "Final Classification",
            prediction_label,
        ],
        [
            "Confidence",
            _percentage(
                payload.get(
                    "confidence"
                )
            ),
        ],
        [
            "Filename",
            str(
                _value(
                    payload,
                    "original_filename",
                    _value(
                        payload,
                        "video",
                    ),
                )
            ),
        ],
    ]

    summary_table = Table(
        summary_data,
        colWidths=[
            50 * mm,
            120 * mm,
        ],
    )

    summary_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.HexColor(
                        "#E8EDF5"
                    ),
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (0, -1),
                    "Helvetica-Bold",
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor(
                        "#B8C2D1"
                    ),
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
            ]
        )
    )

    story.append(summary_table)

    story.append(
        Spacer(
            1,
            10,
        )
    )

    story.append(
        Paragraph(
            interpretation,
            normal_style,
        )
    )

    story.append(
        Paragraph(
            "Probability Analysis",
            section_style,
        )
    )

    probability_data = [
        [
            "Metric",
            "Result",
        ],
        [
            "Real Probability",
            _percentage(
                payload.get(
                    "real_probability"
                )
            ),
        ],
        [
            "Fake Probability",
            _percentage(
                payload.get(
                    "fake_probability"
                )
            ),
        ],
        [
            "Decision Threshold",
            _percentage(
                payload.get(
                    "threshold"
                )
            ),
        ],
    ]

    probability_table = Table(
        probability_data,
        colWidths=[
            85 * mm,
            85 * mm,
        ],
    )

    probability_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#273654"
                    ),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor(
                        "#B8C2D1"
                    ),
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
            ]
        )
    )

    story.append(probability_table)

    story.append(
        Paragraph(
            "Analysis Details",
            section_style,
        )
    )

    details_data = [
        [
            "Frames Analyzed",
            str(
                _value(
                    payload,
                    "frames_analyzed",
                )
            ),
        ],
        [
            "Faces Analyzed",
            str(
                _value(
                    payload,
                    "faces_analyzed",
                )
            ),
        ],
        [
            "Frame Interval",
            str(
                _value(
                    payload,
                    "frame_interval",
                )
            ),
        ],
        [
            "Processing Time",
            (
                f"{_value(payload, 'processing_seconds')} "
                "seconds"
            ),
        ],
        [
            "Video Size",
            _video_size(payload),
        ],
    ]

    details_table = Table(
        details_data,
        colWidths=[
            85 * mm,
            85 * mm,
        ],
    )

    details_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.HexColor(
                        "#F2F4F7"
                    ),
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (0, -1),
                    "Helvetica-Bold",
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor(
                        "#CCCCCC"
                    ),
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
            ]
        )
    )

    story.append(details_table)

    story.append(
        Paragraph(
            "Model Information",
            section_style,
        )
    )

    model_data = [
        [
            "Architecture",
            str(
                _value(
                    payload,
                    "model_name",
                    "EfficientNet-B0",
                )
            ),
        ],
        [
            "Checkpoint",
            str(checkpoint),
        ],
        [
            "System",
            "ADDFS",
        ],
    ]

    model_table = Table(
        model_data,
        colWidths=[
            50 * mm,
            120 * mm,
        ],
    )

    model_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.HexColor(
                        "#F2F4F7"
                    ),
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (0, -1),
                    "Helvetica-Bold",
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor(
                        "#CCCCCC"
                    ),
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
            ]
        )
    )

    story.append(model_table)

    story.append(
        Spacer(
            1,
            18,
        )
    )

    story.append(
        Paragraph(
            (
                "<b>Important:</b> This report represents "
                "the output of an automated machine-learning "
                "system. Classification results should be "
                "treated as decision-support information and "
                "not as absolute proof of authenticity or "
                "manipulation."
            ),
            normal_style,
        )
    )

    document.build(story)

    pdf_bytes = buffer.getvalue()

    buffer.close()

    return pdf_bytes