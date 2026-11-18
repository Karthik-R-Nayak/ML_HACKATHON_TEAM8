"""Generate the comprehensive PathSnap Technical Report PDF with ReportLab."""
import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
PDF_OUT = ROOT / "docs/PathSnap_Technical_Report.pdf"
FIG_DIR = ROOT / "docs/figures"


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))

        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, letter[1] - 36, "PATHSNAP: DISTRIBUTED MAP-MATCHING & ROUTE RECONSTRUCTION — TECHNICAL REPORT")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(54, letter[1] - 42, letter[0] - 54, letter[1] - 42)

        # Footer
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(54, 45, letter[0] - 54, 45)
        self.drawString(54, 32, "PES University • Dept of Computer Science & Engineering • UE24CS352A Machine Learning")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 54, 32, page_str)
        self.restoreState()


def build_pdf():
    doc = SimpleDocTemplate(
        str(PDF_OUT),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()
    primary = colors.HexColor("#0f172a")
    accent = colors.HexColor("#0284c7")
    muted = colors.HexColor("#64748b")
    card_bg = colors.HexColor("#f8fafc")

    title_style = ParagraphStyle(
        "DocTitle", parent=styles["Title"],
        fontName="Helvetica-Bold", fontSize=22, leading=26,
        textColor=primary, alignment=0, spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle", parent=styles["Normal"],
        fontName="Helvetica", fontSize=12, leading=16,
        textColor=accent, spaceAfter=14,
    )
    h1_style = ParagraphStyle(
        "H1", parent=styles["Heading1"],
        fontName="Helvetica-Bold", fontSize=14, leading=18,
        textColor=primary, spaceBefore=14, spaceAfter=8,
    )
    h2_style = ParagraphStyle(
        "H2", parent=styles["Heading2"],
        fontName="Helvetica-Bold", fontSize=11, leading=15,
        textColor=accent, spaceBefore=10, spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontName="Helvetica", fontSize=9.5, leading=13.5,
        textColor=colors.HexColor("#334155"), spaceAfter=6,
    )
    bullet_style = ParagraphStyle(
        "Bullet", parent=body_style,
        leftIndent=15, bulletIndent=6, spaceAfter=3,
    )
    code_style = ParagraphStyle(
        "Code", parent=styles["Normal"],
        fontName="Courier", fontSize=8, leading=10,
        textColor=colors.HexColor("#0f172a"), backColor=colors.HexColor("#f1f5f9"),
        borderPadding=4, spaceAfter=6,
    )

    story = []

    # Title & Metadata Block
    story.append(Paragraph("PATHSNAP: DISTRIBUTED MAP-MATCHING & ROUTE RECONSTRUCTION", title_style))
    story.append(Paragraph("Comprehensive Technical Report • System Architecture, ML Model Selection & HMM Route Reconstruction", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=accent, spaceAfter=12))

    meta_data = [
        [Paragraph("<b>Course:</b> UE24CS352A Machine Learning", body_style),
         Paragraph("<b>Institution:</b> PES University, RR Campus", body_style)],
        [Paragraph("<b>Architecture:</b> 2-Laptop Distributed Deployment", body_style),
         Paragraph("<b>Selected Classifier:</b> Random Forest (120 trees, depth 18)", body_style)],
        [Paragraph("<b>Route Decoder:</b> Directed HMM / Viterbi Dynamic Prog.", body_style),
         Paragraph("<b>Mean Student HMM Accuracy:</b> <b>90.00%</b> (+2.19% over Top-1)", body_style)],
    ]
    t_meta = Table(meta_data, colWidths=[250, 254])
    t_meta.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), card_bg),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 10))

    # Section 1: Executive Summary & Architecture
    story.append(Paragraph("1. Executive Summary & Distributed Deployment Architecture", h1_style))
    story.append(Paragraph(
        "PathSnap resolves the critical real-world problem of GPS inaccuracies in urban environments—where multipath reflections, tall structures, atmospheric noise, and low sampling frequencies cause vehicles to appear to jump across buildings or parallel streets. We developed an end-to-end distributed system that reconstructs coherent, topologically valid vehicle trajectories by coupling offline machine learning classification with Hidden Markov Model (HMM) Viterbi route decoding.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Two-Laptop Distributed Deployment Model:</b>", body_style
    ))
    story.append(Paragraph("• <b>Laptop 1 (Model & API):</b> Hosts the trained classifier pipeline, road-network graph topology, Member 3 HMM/Viterbi route decoder, and FastAPI REST endpoints. Computes server-side validation against ground-truth routes without exposing labels.", bullet_style))
    story.append(Paragraph("• <b>Laptop 2 (Evaluator Interface):</b> Evaluator-facing client running on port 5173 / browser. Communicates over the local network to Laptop 1, retrieves available evaluation trips, requests full route reconstruction, and renders interactive multi-layer Leaflet maps with performance analytics.", bullet_style))
    story.append(Spacer(1, 8))

    # Section 2: Dataset Understanding & EDA
    story.append(Paragraph("2. Dataset Understanding and Exploratory Data Analysis (EDA)", h1_style))
    story.append(Paragraph(
        "The benchmark dataset consists of 80 training trips, 10 student-testing trips, and 10 reserved evaluator-testing trips. Each trip folder provides four raw, whitespace-separated text files: <code>.track</code> (GPS observations: lon, lat, timestamp), <code>.nodes</code> (road network vertices: lon, lat), <code>.arcs</code> (directed road connections: from_node, to_node), and <code>.route</code> (chronological reference ground truth arc IDs).",
        body_style
    ))
    story.append(Paragraph("<b>Key Exploratory Findings:</b>", body_style))
    story.append(Paragraph("• <b>Sampling Characteristics:</b> Observations are nominally 1 Hz (97.2% of consecutive steps have exact Δt = 1.0 s). However, occasional large temporal gaps (Δt > 15 s) exist due to signal loss and tunnels.", bullet_style))
    story.append(Paragraph("• <b>Candidate Generation:</b> Member 1 generated candidates using a 100 m search radius, maximum 30 candidates per observation, indexed via a 100 m spatial grid. Across 10 student trips, 366,633 candidate-observation pairs were generated.", bullet_style))
    story.append(Paragraph("• <b>17 Ordered Features:</b> Includes distance to arc (<code>dist_m</code>), projected fraction along segment (<code>proj_frac</code>), heading difference (<code>heading_diff_deg</code>), GPS speed & bearing, arc bearing & length, topological node in/out degrees, and local distance rankings.", bullet_style))
    story.append(Spacer(1, 8))

    # Section 3: Model Experimentation & Comparison
    story.append(Paragraph("3. Model Experimentation and 5-Model Comparison", h1_style))
    story.append(Paragraph(
        "In strict accordance with the hackathon requirements, five distinct model families were trained and evaluated using trip-grouped validation splits (GroupKFold with 16 internal holdout trips and 64 cross-validation trips).",
        body_style
    ))

    model_table_data = [
        ["Model", "Algorithm Family", "Key Configuration", "CV Macro F1", "CV Top-1", "Student Top-1", "Route Cont."],
        ["Logistic Reg.", "Linear Probabilistic", "L2 penalty, liblinear, C=1.0", "89.82%", "82.59%", "86.22%", "88.4%"],
        ["KNN", "Instance-Based", "k=15, distance-weighted, 100 obs/trip", "85.12%", "76.42%", "81.93%", "82.1%"],
        ["Decision Tree", "Non-linear Tree", "max_depth=14, min_samples_leaf=20", "89.45%", "82.10%", "86.85%", "89.3%"],
        ["Random Forest (Sel.)", "Ensemble Bagging", "120 trees, max_depth=18, min_leaf=10", "90.61%", "83.40%", "87.81%", "92.1%"],
        ["Hist Gradient Boost", "Boosting", "120 trees, lr=0.08, max_bins=255", "90.44%", "83.16%", "87.94%", "91.8%"],
    ]
    t_models = Table(model_table_data, colWidths=[100, 85, 125, 55, 48, 54, 45])
    t_models.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), primary),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (3, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, card_bg]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_models)
    story.append(Spacer(1, 8))

    if (FIG_DIR / "01_model_comparison.png").is_file():
        story.append(Image(str(FIG_DIR / "01_model_comparison.png"), width=6.8 * inch, height=2.1 * inch))
        story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Model Selection Justification:</b> Random Forest was selected because it achieved the highest cross-validation Macro F1 (90.61%) and highest CV Top-1 Accuracy (83.40%), with superior generalisation across out-of-fold trips and resilience against noisy GPS outliers.", body_style))
    story.append(Spacer(1, 8))

    # Page Break for HMM Section
    story.append(PageBreak())

    # Section 4: Bayesian Optimisation
    story.append(Paragraph("4. Bayesian Hyperparameter Optimisation", h1_style))
    story.append(Paragraph(
        "Bayesian optimisation was performed using Optuna Tree-structured Parzen Estimators (TPE) over 12 trials on Histogram Gradient Boosting. Searched parameters: learning rate [0.01, 0.2], max_leaf_nodes [15, 63], min_samples_leaf [10, 50], and L2 regularization [1e-4, 10.0]. CV Top-1 marginally improved from 83.16% to 83.32%, but did not surpass Random Forest's 83.40% CV Top-1. Following strict frozen-selection policy, untuned Random Forest was frozen as the authoritative production artifact.",
        body_style
    ))
    if (FIG_DIR / "03_bayesian_optimization.png").is_file():
        story.append(Image(str(FIG_DIR / "03_bayesian_optimization.png"), width=5.5 * inch, height=2.3 * inch))
        story.append(Spacer(1, 6))

    # Section 5: Member 3 HMM and Viterbi Route Reconstruction
    story.append(Paragraph("5. Member 3 HMM and Viterbi Route Reconstruction", h1_style))
    story.append(Paragraph(
        "Candidate classification treats each observation independently. To eliminate zigzagging and topological disconnections, Member 3 implements an HMM decoded via Viterbi dynamic programming:",
        body_style
    ))
    story.append(Paragraph("• <b>Hidden States:</b> Candidate road segments <i>c_{t, i}</i> for each observation <i>t</i>.", bullet_style))
    story.append(Paragraph("• <b>Emission Probabilities:</b> Log candidate posteriors from Member 2 Random Forest: log(max(p_{t, i}, 10^{-6})).", bullet_style))
    story.append(Paragraph("• <b>Transition Probabilities:</b> Modeled as an exponential penalty on distance discrepancy: log P(s_t | s_{t-1}) = -|d_{net} - d_{gps}| / β, where β = 10.0 m. Network shortest paths are computed across directed road network topology using <code>scipy.sparse.csgraph.dijkstra</code>.", bullet_style))
    story.append(Paragraph("• <b>Gap & Disconnection Handling:</b> Time gaps Δt > 15 s trigger an HMM chain reset to avoid erroneous leaps across missing data; impossible transitions trigger a soft penalty fallback.", bullet_style))
    story.append(Spacer(1, 6))

    # Benchmark Table
    story.append(Paragraph("<b>Benchmark Across All 10 Student Evaluation Trips:</b>", body_style))
    hmm_table_data = [
        ["Trip ID", "Obs Count", "Top-1 Accuracy", "HMM Accuracy", "Top-1 Continuity", "HMM Continuity", "Latency"],
        ["00000000", "2,503", "93.97%", "95.96%", "96.24%", "99.60%", "1.61 s"],
        ["00000007", "1,354", "93.87%", "88.63%", "99.41%", "99.85%", "0.63 s"],
        ["00000009", "1,049", "86.37%", "91.80%", "87.79%", "95.04%", "1.97 s"],
        ["00000040", "2,995", "91.95%", "92.59%", "96.16%", "99.40%", "1.77 s"],
        ["00000052", "262", "95.42%", "100.00%", "95.02%", "96.55%", "0.30 s"],
        ["00000056", "3,958", "88.65%", "93.10%", "88.43%", "96.54%", "4.61 s"],
        ["00000065", "768", "86.85%", "88.67%", "91.40%", "98.17%", "1.15 s"],
        ["00000076", "1,486", "87.08%", "92.60%", "87.81%", "96.30%", "2.47 s"],
        ["00000077", "3,911", "68.78%", "70.01%", "91.64%", "97.85%", "4.85 s"],
        ["00000091", "874", "85.13%", "86.61%", "87.51%", "96.56%", "1.20 s"],
        ["Mean", "1,916", "87.81%", "90.00% (+2.19%)", "92.14%", "97.59% (+5.45%)", "2.05 s"],
    ]
    t_hmm = Table(hmm_table_data, colWidths=[65, 58, 75, 80, 80, 80, 54])
    t_hmm.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), primary),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.white, card_bg]),
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#e0f2fe")),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t_hmm)
    story.append(Spacer(1, 6))

    if (FIG_DIR / "02_hmm_vs_top1_student_trips.png").is_file():
        story.append(Image(str(FIG_DIR / "02_hmm_vs_top1_student_trips.png"), width=6.8 * inch, height=3.0 * inch))
        story.append(Spacer(1, 6))

    # Page Break for Folium and Deployment Section
    story.append(PageBreak())

    # Section 6: Folium Geographical Visualisations
    story.append(Paragraph("6. Geographical Visualisation Using Folium", h1_style))
    story.append(Paragraph(
        "Interactive Folium maps were generated for evaluation trips (e.g. <code>trip_00000052_folium_map.html</code>, <code>trip_00000000_folium_map.html</code>). The map incorporates four distinct layers accessible via Folium's LayerControl:",
        body_style
    ))
    story.append(Paragraph("1. <b>Raw GPS Track (Blue circles):</b> Visualises point-level GPS scatter and sampling jitter.", bullet_style))
    story.append(Paragraph("2. <b>Road Network Candidates (Gray thin lines):</b> All road segments queried within the 100m radius.", bullet_style))
    story.append(Paragraph("3. <b>Reference Route (Purple dashed line):</b> Ground-truth sequence of traversed arcs.", bullet_style))
    story.append(Paragraph("4. <b>PathSnap Matched Route (Emerald green bold line):</b> The reconstructed trajectory from HMM/Viterbi.", bullet_style))
    story.append(Spacer(1, 6))

    # Section 7: API, UI & Deployment
    story.append(Paragraph("7. API, UI and Two-Laptop Deployment Verification", h1_style))
    story.append(Paragraph(
        "The complete system is configured for live two-laptop demonstration between Laptop 1 and Laptop 2:",
        body_style
    ))

    story.append(Paragraph("<b>FastAPI Endpoints on Laptop 1:</b>", body_style))
    api_table_data = [
        ["Method", "Endpoint", "Function"],
        ["GET", "/healthz", "Returns model readiness, SHA-256 digest, and HMM decoder status"],
        ["GET", "/v1/pathsnap/capabilities", "Reports readiness of all 5 stages (preprocessing -> ML -> HMM)"],
        ["GET", "/v1/pathsnap/trips", "Lists available evaluation trips"],
        ["POST", "/v1/pathsnap/trips/{trip_id}/match", "Full end-to-end pipeline + server-side ground truth metrics"],
        ["POST", "/v1/pathsnap/match", "Arbitrary GPS trajectory JSON matching against custom road network"],
        ["GET", "/ui", "Serves the Evaluator Dashboard with interactive Leaflet map"],
    ]
    t_api = Table(api_table_data, colWidths=[55, 175, 262])
    t_api.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), accent),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, card_bg]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_api)
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>Startup & Execution Instructions:</b>", body_style))
    story.append(Paragraph("<b>On Laptop 1 (Model & Server):</b>", body_style))
    story.append(Paragraph(
        "cd pathsnap-repo<br/>"
        "& \"..\\.venv\\Scripts\\python.exe\" -m uvicorn laptop1_demo.server:app --host 0.0.0.0 --port 8000",
        code_style
    ))
    story.append(Paragraph("<b>On Laptop 2 (Evaluator UI Client):</b>", body_style))
    story.append(Paragraph(
        "python -m http.server 5173 --bind 127.0.0.1<br/>"
        "# Open http://localhost:5173 in browser, enter http://10.0.13.39:8000, and click Connect & Match!",
        code_style
    ))
    story.append(Paragraph("<i>Alternative Direct Access:</i> Laptop 2 can also directly navigate to <code>http://10.0.13.39:8000/ui</code> to access the full interactive interface without launching a local server.", body_style))

    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Academic Integrity & Independence Statement:</b> All models, preprocessing pipelines, Bayesian trials, HMM decoders, API services, and user interfaces were developed independently from the raw data. No evaluator labels were used during model training or hyperparameter selection.", ParagraphStyle("Notice", parent=body_style, fontName="Helvetica-Oblique", textColor=muted)))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Technical Report built successfully: {PDF_OUT}")


if __name__ == "__main__":
    build_pdf()
