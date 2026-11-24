"""Downloadable analysis report: PDF (reportlab) with an HTML fallback if reportlab is not installed."""
import io
from html import escape

from . import backend as B

DISCLAIMER = ("This is a machine-learning based assessment trained on a research dataset. It is not a guarantee that a "
              "transaction is safe or fraudulent. Always verify payment details yourself.")


def _rows(res, amount):
    rows = [("Amount entered (context only)", f"Rs. {amount:,.2f}")]
    for c, v in res["inputs"].items():
        rows.append((B.pretty(c), B.value_text(c, v) + ("  (typical value assumed)" if c in res["assumed"] else "")))
    return rows


def build(res, amount):
    """Returns (bytes, mime, filename)."""
    verdict = "Potential fraud detected" if res["pred"] == 1 else "Transaction appears legitimate"
    expl = B.explanation_text(res)
    tips = B.safety_tips(res, res["inputs"])
    factors = [(d["label"], d["vtext"], f"{d['shap']*100:+.1f} pts", ("no real effect" if abs(d["shap"]) < 0.01 else "towards fraud" if d["shap"] > 0 else "towards genuine"))
               for d in res["contrib"]]
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError:
        h = ["<html><body style='font-family:sans-serif;max-width:760px;margin:30px auto'>",
             "<h1>UPI Transaction Analysis Report</h1>", f"<p>{res['time']}</p>", f"<h2>{escape(verdict)}</h2>",
             f"<p>Model fraud probability: <b>{res['prob']*100:.1f}%</b> &middot; Risk level: {res['risk']}</p>",
             "<h3>Details entered</h3><table border=1 cellpadding=6>"]
        h += [f"<tr><td>{escape(a)}</td><td>{escape(b)}</td></tr>" for a, b in _rows(res, amount)]
        h += ["</table><h3>Factors</h3><table border=1 cellpadding=6>"]
        h += [f"<tr><td>{escape(a)}</td><td>{escape(b)}</td><td>{c}</td><td>{d}</td></tr>" for a, b, c, d in factors]
        h += ["</table><h3>Explanation</h3>"] + [f"<p>{escape(x)}</p>" for x in expl]
        h += ["<h3>Safety recommendations</h3><ul>"] + [f"<li>{escape(t)}</li>" for t in tips]
        h += [f"</ul><p><i>{DISCLAIMER}</i></p></body></html>"]
        return "\n".join(h).encode(), "text/html", "upi_transaction_report.html"

    ss = getSampleStyleSheet()
    H = ParagraphStyle("H", parent=ss["Title"], textColor=colors.HexColor("#3730A3"), fontSize=20, alignment=0)
    H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=colors.HexColor("#3730A3"), spaceBefore=10)
    body = ss["BodyText"]
    bad = res["pred"] == 1
    VS = ParagraphStyle("V", parent=ss["Heading1"], fontSize=17,
                        textColor=colors.HexColor("#B91C1C" if bad else "#047857"))
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm)
    el = [Paragraph("UPI Transaction Analysis Report", H), Paragraph(f"Generated {res['time']}", body), Spacer(1, 8),
          Paragraph(verdict, VS),
          Paragraph(f"Model fraud probability: <b>{res['prob']*100:.1f}%</b> &nbsp;|&nbsp; Risk level: <b>{res['risk']}</b> "
                    f"&nbsp;|&nbsp; Model: {B.artifacts()['info']['B']['model_name']}", body)]

    def table(data, widths, head=None):
        d = ([head] if head else []) + [list(map(str, r)) for r in data]
        t = Table(d, colWidths=widths, repeatRows=1 if head else 0)
        st_ = [("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#C7D2FE")), ("FONTSIZE", (0, 0), (-1, -1), 9),
               ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
        if head:
            st_ += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E0E7FF")), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
        t.setStyle(TableStyle(st_))
        return t

    el += [Paragraph("Details entered", H2), table(_rows(res, amount), [90 * mm, 80 * mm]),
           Paragraph("What influenced the result", H2),
           table(factors, [62 * mm, 28 * mm, 30 * mm, 50 * mm], ["Factor", "Your value", "Effect", "Direction"])]
    el += [Paragraph("In simple words", H2)] + [Paragraph(escape(x), body) for x in expl]
    el += [Paragraph("Safety recommendations", H2)] + [Paragraph("&bull; " + escape(t), body) for t in tips]
    el += [Spacer(1, 10), Paragraph(f"<i>{DISCLAIMER}</i>", body)]
    doc.build(el)
    return buf.getvalue(), "application/pdf", "upi_transaction_report.pdf"
