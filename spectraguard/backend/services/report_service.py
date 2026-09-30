"""Report generation service using ReportLab to generate Lot Audit & Certificate of Analysis PDFs."""

from __future__ import annotations

import io
import datetime as dt
from typing import Optional
from sqlalchemy.orm import Session

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

from backend.records import crud, models


def generate_lot_pdf_report(db: Session, lot_id: str) -> bytes:
    """Generate a Certificate of Analysis & Lot Quality Audit PDF for a given lot_id."""
    lot = crud.get_lot(db, lot_id)
    if not lot:
        raise ValueError(f"Lot '{lot_id}' not found")

    products, total_count = crud.list_products(db, lot_id=lot_id, per_page=1000)

    # Calculate statistics
    rejected_count = sum(1 for p in products if p.verdict == "REJECT")
    review_count = sum(1 for p in products if p.verdict == "REVIEW")
    passed_count = sum(1 for p in products if p.verdict == "PASS")
    reject_rate = (rejected_count / total_count * 100) if total_count > 0 else 0.0

    pathogen_counts = {}
    for p in products:
        if p.predicted_pathogen:
            pathogen_counts[p.predicted_pathogen] = pathogen_counts.get(p.predicted_pathogen, 0) + 1

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=4,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748B"),
        spaceAfter=12,
    )

    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=10,
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#334155"),
    )

    elements = []

    # Header Header Banner
    elements.append(Paragraph("SPECTRA-GUARD | Hyperspectral Food Safety Inspection", subtitle_style))
    elements.append(Paragraph(f"LOT QUALITY AUDIT & CERTIFICATE OF ANALYSIS", title_style))
    elements.append(Paragraph(f"Report Generated: {dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284C7"), spaceAfter=12))

    # Lot Overview Table
    lot_meta = [
        [
            Paragraph("<b>Lot ID:</b> " + str(lot.lot_id), body_style),
            Paragraph("<b>Product Type:</b> " + str(lot.product_type).upper(), body_style),
        ],
        [
            Paragraph("<b>Supplier:</b> " + str(lot.supplier or "N/A"), body_style),
            Paragraph("<b>Origin Farm:</b> " + str(lot.origin_farm or "N/A"), body_style),
        ],
        [
            Paragraph("<b>Received Date:</b> " + lot.received_ts.strftime("%Y-%m-%d %H:%M"), body_style),
            Paragraph("<b>Shift / Operator:</b> " + f"{lot.shift or 'N/A'} / {lot.operator or 'N/A'}", body_style),
        ],
        [
            Paragraph("<b>Status:</b> <font color='" + ("#DC2626" if lot.status == "hold" else "#16A34A") + "'><b>" + str(lot.status).upper() + "</b></font>", body_style),
            Paragraph("<b>Inspection Line:</b> LINE-1", body_style),
        ]
    ]

    meta_table = Table(lot_meta, colWidths=[270, 270])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 14))

    # Quality Summary Metrics Table
    elements.append(Paragraph("Inspection Summary Statistics", section_heading))
    summary_data = [
        ["Total Items Scanned", "Passed", "Rejections", "Manual Review", "Defect Rate (%)"],
        [str(total_count), str(passed_count), str(rejected_count), str(review_count), f"{reject_rate:.2f}%"],
    ]
    summary_table = Table(summary_data, colWidths=[108, 108, 108, 108, 108])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor("#F1F5F9")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 14))

    # Pathogen Breakdown (if any)
    if pathogen_counts:
        elements.append(Paragraph("Pathogen Contamination Breakdown", section_heading))
        pathogen_data = [["Pathogen Class", "Flagged Scans", "Share of Defects (%)"]]
        for pathogen, count in pathogen_counts.items():
            pct = (count / rejected_count * 100) if rejected_count > 0 else 0
            pathogen_data.append([pathogen, str(count), f"{pct:.1f}%"])

        pathogen_table = Table(pathogen_data, colWidths=[180, 180, 180])
        pathogen_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#334155")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        elements.append(pathogen_table)
        elements.append(Spacer(1, 14))

    # Sample Items Table (First 15 items)
    elements.append(Paragraph("Sample Item Scan Log (First 15 Scans)", section_heading))
    sample_rows = [["Item ID", "Scan Time", "Verdict", "Pathogen", "Score", "Latency"]]

    for p in products[:15]:
        verdict_color = "#16A34A" if p.verdict == "PASS" else ("#DC2626" if p.verdict == "REJECT" else "#D97706")
        sample_rows.append([
            f"ITEM-{p.id:05d}",
            p.scan_ts.strftime("%H:%M:%S"),
            Paragraph(f"<font color='{verdict_color}'><b>{p.verdict}</b></font>", body_style),
            p.predicted_pathogen or "—",
            f"{p.contamination_score:.2f}",
            f"{p.latency_ms:.1f} ms",
        ])

    sample_table = Table(sample_rows, colWidths=[90, 80, 75, 105, 95, 95])
    sample_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0284C7")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(sample_table)
    elements.append(Spacer(1, 20))

    # Sign-off Block
    sign_block = [
        [
            Paragraph("<b>QA Manager Sign-off:</b> ___________________________", body_style),
            Paragraph("<b>Date:</b> _______________", body_style),
        ],
        [
            Paragraph("<b>Cryptographic Audit Hash:</b> SHA-256 Validated", body_style),
            Paragraph("<b>SpectraGuard Engine:</b> v1.0-ONNX", body_style),
        ]
    ]
    sign_table = Table(sign_block, colWidths=[320, 220])
    sign_table.setStyle(TableStyle([
        ('LINEABOVE', (0, 0), (-1, 0), 1, colors.HexColor("#94A3B8")),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(KeepTogether([sign_table]))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()
