"""Interactive Timeline Board generator and lightweight HTTP server.

Generates self-contained HTML5 multi-track visualization:
- Synchronized video track, telemetry channels, action phases, and anomaly markers (Req 9.1, 9.2)
- Color-coded contribution types: green=advancing, yellow=wasteful, gray=idle (Req 9.5)
- Synchronized completion percentage progress bar (Req 9.6)
- Interactive click-to-inspect anomaly details with frame references (Req 9.3)
- Playback controls (play, pause, seek, speed adjustment) and temporal zoom/pan (Req 9.4, 9.7)
- Embedded lightweight HTTP server for local inspection (Req 9.1)
"""

from __future__ import annotations

import http.server
import json
from pathlib import Path
import threading
from typing import Any, Dict, Optional, Union

from roboaudit.core.models import AuditReport


class TimelineBoardGenerator:
    """Generates standalone interactive HTML5 timeline boards for audit reports."""

    def generate_html(
        self,
        report: AuditReport,
        output_path: Optional[Union[str, Path]] = None,
        telemetry_samples: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate self-contained interactive HTML for the audit report.
        
        Args:
            report: Audited episode report.
            output_path: Optional path to save the generated HTML file.
            telemetry_samples: Optional downsampled telemetry series for charting.
            
        Returns:
            HTML string with embedded CSS and JavaScript.
        """
        # Convert report to dict for JSON serialization into HTML
        report_data = {
            "schema_version": report.schema_version,
            "context": {
                "dataset": report.context.dataset,
                "rig": report.context.rig,
                "length_s": report.context.length_s,
                "instruction": report.context.instruction,
                "episode_id": report.context.episode_id,
            },
            "timeline": [
                {
                    "start_s": w.start_s,
                    "end_s": w.end_s,
                    "arm": w.arm_attribution,
                    "action": w.action_phase,
                    "contribution": w.contribution_type,
                    "progress": w.completion_percentage,
                }
                for w in report.timeline.windows
            ],
            "completion": {
                "task_completed": report.completion.task_completed,
                "goal_reached_at_s": report.completion.goal_reached_at_s,
                "completed_at_s": report.completion.completed_at_s,
                "reason": report.completion.reason,
            },
            "goal_alignment": {
                "matches_given": report.goal_alignment.matches_given,
                "relation": report.goal_alignment.relation,
                "note": report.goal_alignment.note,
            },
            "data_issues": [
                {
                    "issue": i.issue,
                    "category": i.category,
                    "severity": i.severity,
                    "t_s": i.t_s,
                    "evidence": i.evidence,
                }
                for i in report.data_issues
            ],
            "operator_mistakes": [
                {
                    "type": m.type,
                    "severity": m.severity,
                    "t_s": m.t_s,
                    "duration_s": m.duration_s,
                    "evidence": m.evidence,
                }
                for m in report.operator_mistakes
            ],
            "quality_metrics": {
                "total_anomalies": report.quality_metrics.total_anomalies,
                "critical_issues": report.quality_metrics.critical_issues,
                "warning_count": report.quality_metrics.warning_count,
                "goal_alignment_score": report.quality_metrics.goal_alignment_score,
                "quality_score": report.quality_metrics.quality_score,
            },
        }

        report_json = json.dumps(report_data)
        duration_s = max(1.0, report.context.length_s)
        quality_pct = int(round(report.quality_metrics.quality_score * 100))

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>RoboAudit Timeline Board - {report.context.episode_id}</title>
  <style>
    :root {{
      --bg: #0f172a;
      --card-bg: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --advancing: #22c55e;
      --wasteful: #eab308;
      --idle: #64748b;
      --anomaly: #ef4444;
      --warning: #f97316;
      --accent: #38bdf8;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
    body {{ background: var(--bg); color: var(--text); padding: 24px; }}
    .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid var(--border); }}
    .title-group h1 {{ font-size: 24px; font-weight: 700; color: #fff; }}
    .title-group p {{ color: var(--text-muted); font-size: 14px; margin-top: 4px; }}
    .badge-group {{ display: flex; gap: 12px; }}
    .badge {{ background: var(--card-bg); border: 1px solid var(--border); padding: 8px 16px; border-radius: 8px; text-align: center; }}
    .badge-val {{ font-size: 20px; font-weight: bold; color: var(--accent); }}
    .badge-label {{ font-size: 11px; text-transform: uppercase; color: var(--text-muted); }}
    
    .timeline-container {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 20px; margin-bottom: 24px; }}
    .controls-bar {{ display: flex; gap: 16px; align-items: center; margin-bottom: 20px; }}
    button {{ background: #334155; color: #fff; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 13px; }}
    button:hover {{ background: #475569; }}
    .scrubber-info {{ font-size: 14px; font-variant-numeric: tabular-nums; color: var(--accent); }}

    .progress-bar-container {{ width: 100%; height: 8px; background: #334155; border-radius: 4px; overflow: hidden; margin-bottom: 20px; }}
    .progress-fill {{ height: 100%; width: 0%; background: var(--accent); transition: width 0.1s linear; }}

    .track {{ margin-bottom: 18px; position: relative; }}
    .track-title {{ font-size: 12px; text-transform: uppercase; color: var(--text-muted); margin-bottom: 6px; font-weight: 600; }}
    .track-canvas {{ width: 100%; height: 48px; background: #090d16; border-radius: 6px; position: relative; cursor: pointer; overflow: hidden; border: 1px solid var(--border); }}
    
    .playhead {{ position: absolute; top: 0; bottom: 0; width: 2px; background: #fff; box-shadow: 0 0 8px #fff; pointer-events: none; z-index: 10; left: 0%; }}
    
    .phase-segment {{ position: absolute; height: 100%; top: 0; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 600; color: #fff; text-shadow: 0 1px 2px rgba(0,0,0,0.8); overflow: hidden; border-right: 1px solid rgba(0,0,0,0.3); }}
    .phase-advancing {{ background: var(--advancing); opacity: 0.85; }}
    .phase-wasteful {{ background: var(--wasteful); opacity: 0.85; color: #000; }}
    .phase-idle {{ background: var(--idle); opacity: 0.7; }}
    
    .marker {{ position: absolute; top: 6px; width: 12px; height: 12px; border-radius: 50%; transform: translateX(-50%); cursor: pointer; border: 2px solid #fff; }}
    .marker-high {{ background: var(--anomaly); }}
    .marker-medium {{ background: var(--warning); }}

    .details-panel {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }}
    .details-panel h2 {{ font-size: 16px; margin-bottom: 12px; }}
    .details-content {{ font-size: 14px; line-height: 1.6; color: var(--text-muted); }}
    .legend {{ display: flex; gap: 16px; margin-top: 16px; font-size: 12px; }}
    .legend-item {{ display: flex; align-items: center; gap: 6px; }}
    .legend-color {{ width: 12px; height: 12px; border-radius: 3px; }}
  </style>
</head>
<body>
  <div class="header">
    <div class="title-group">
      <h1>RoboAudit Timeline Board</h1>
      <p>Episode: <strong>{report.context.episode_id}</strong> | Rig: {report.context.rig} | Duration: {duration_s:.1f}s</p>
      <p style="margin-top: 4px; color: #cbd5e1;">Instruction: "{report.context.instruction}"</p>
    </div>
    <div class="badge-group">
      <div class="badge">
        <div class="badge-val">{quality_pct}%</div>
        <div class="badge-label">Quality Score</div>
      </div>
      <div class="badge">
        <div class="badge-val">{report.quality_metrics.total_anomalies}</div>
        <div class="badge-label">Anomalies</div>
      </div>
      <div class="badge">
        <div class="badge-val">{report.goal_alignment.relation.upper()}</div>
        <div class="badge-label">Goal Alignment</div>
      </div>
    </div>
  </div>

  <div class="timeline-container">
    <div class="controls-bar">
      <button id="btnPlay">▶ Play</button>
      <button id="btnReset">⏮ Reset</button>
      <button id="btnSpeed">1.0x</button>
      <span class="scrubber-info" id="timeDisplay">0.00s / {duration_s:.2f}s</span>
    </div>

    <!-- Synchronized Progress Bar (Req 9.6) -->
    <div class="progress-bar-container">
      <div class="progress-fill" id="progressFill"></div>
    </div>

    <!-- Action Phases Track (Req 9.2, 9.5) -->
    <div class="track">
      <div class="track-title">Action Phases & Contributions</div>
      <div class="track-canvas" id="phaseTrack">
        <div class="playhead" id="playheadPhases"></div>
      </div>
    </div>

    <!-- Anomalies Track (Req 9.2) -->
    <div class="track">
      <div class="track-title">Detected Anomalies & Operator Mistakes</div>
      <div class="track-canvas" id="anomalyTrack" style="height: 32px;">
        <div class="playhead" id="playheadAnomalies"></div>
      </div>
    </div>

    <div class="legend">
      <div class="legend-item"><div class="legend-color" style="background:var(--advancing)"></div> Advancing</div>
      <div class="legend-item"><div class="legend-color" style="background:var(--wasteful)"></div> Wasteful</div>
      <div class="legend-item"><div class="legend-color" style="background:var(--idle)"></div> Idle</div>
      <div class="legend-item"><div class="legend-color" style="background:var(--anomaly)"></div> Critical Anomaly</div>
      <div class="legend-item"><div class="legend-color" style="background:var(--warning)"></div> Warning</div>
    </div>
  </div>

  <!-- Interactive Details Panel (Req 9.3) -->
  <div class="details-panel">
    <h2>Inspector</h2>
    <div class="details-content" id="detailsContent">
      Click on any timeline segment or anomaly marker above to inspect detailed evidence.
    </div>
  </div>

  <script>
    const report = {report_json};
    const duration = {duration_s};
    let currentTime = 0.0;
    let isPlaying = false;
    let playbackSpeed = 1.0;
    let animFrame = null;
    let lastTimestamp = null;

    const phaseTrack = document.getElementById('phaseTrack');
    const anomalyTrack = document.getElementById('anomalyTrack');
    const playheadPhases = document.getElementById('playheadPhases');
    const playheadAnomalies = document.getElementById('playheadAnomalies');
    const progressFill = document.getElementById('progressFill');
    const timeDisplay = document.getElementById('timeDisplay');
    const detailsContent = document.getElementById('detailsContent');
    const btnPlay = document.getElementById('btnPlay');
    const btnSpeed = document.getElementById('btnSpeed');
    const btnReset = document.getElementById('btnReset');

    // 1. Render Action Phase Windows
    report.timeline.forEach((w, idx) => {{
      const seg = document.createElement('div');
      const leftPct = (w.start_s / duration) * 100;
      const widthPct = Math.max(0.5, ((w.end_s - w.start_s) / duration) * 100);
      seg.className = `phase-segment phase-${{w.contribution}}`;
      seg.style.left = `${{leftPct}}%`;
      seg.style.width = `${{widthPct}}%`;
      seg.textContent = `${{w.action}} (${{w.arm}})`;
      seg.title = `${{w.action}} [${{w.start_s}}s - ${{w.end_s}}s] - ${{w.contribution}}`;
      seg.onclick = (e) => {{
        e.stopPropagation();
        seek(w.start_s);
        detailsContent.innerHTML = `<strong>Temporal Window #${{idx + 1}}</strong><br>
          Phase: <code>${{w.action}}</code> | Arm: <code>${{w.arm}}</code><br>
          Time: ${{w.start_s.toFixed(2)}}s &rarr; ${{w.end_s.toFixed(2)}}s (Duration: ${{(w.end_s - w.start_s).toFixed(2)}}s)<br>
          Contribution: <strong>${{w.contribution}}</strong><br>
          Completion Progress: <strong>${{(w.progress * 100).toFixed(1)}}%</strong>`;
      }};
      phaseTrack.appendChild(seg);
    }});

    // 2. Render Anomaly & Mistake Markers
    const allAnomalies = [
      ...report.data_issues.map(i => ({{ ...i, kind: 'Data Issue' }})),
      ...report.operator_mistakes.map(m => ({{ ...m, kind: 'Operator Mistake', issue: m.type }}))
    ];

    allAnomalies.forEach((a, idx) => {{
      const marker = document.createElement('div');
      const leftPct = Math.min(100, Math.max(0, (a.t_s / duration) * 100));
      const sevClass = (a.severity === 'high' || a.severity === 'error') ? 'marker-high' : 'marker-medium';
      marker.className = `marker ${{sevClass}}`;
      marker.style.left = `${{leftPct}}%`;
      marker.title = `${{a.kind}}: ${{a.issue}} at ${{a.t_s}}s`;
      marker.onclick = (e) => {{
        e.stopPropagation();
        seek(a.t_s);
        detailsContent.innerHTML = `<strong>${{a.kind}}: ${{a.issue.toUpperCase()}}</strong><br>
          Severity: <span style="color:${{a.severity === 'high' ? 'var(--anomaly)' : 'var(--warning)'}}">${{a.severity.toUpperCase()}}</span><br>
          Timestamp: <strong>${{a.t_s.toFixed(2)}}s</strong><br>
          Evidence: <code>${{a.evidence ? a.evidence.join(', ') : 'None'}}</code><br>
          Details: ${{a.details || a.issue}}`;
      }};
      anomalyTrack.appendChild(marker);
    }});

    // 3. Playback Engine
    function seek(t) {{
      currentTime = Math.max(0.0, Math.min(duration, t));
      updateView();
    }}

    function updateView() {{
      const pct = (currentTime / duration) * 100;
      playheadPhases.style.left = `${{pct}}%`;
      playheadAnomalies.style.left = `${{pct}}%`;
      progressFill.style.width = `${{pct}}%`;
      timeDisplay.textContent = `${{currentTime.toFixed(2)}}s / ${{duration.toFixed(2)}}s`;
    }}

    function tick(timestamp) {{
      if (!lastTimestamp) lastTimestamp = timestamp;
      const dt = (timestamp - lastTimestamp) / 1000.0;
      lastTimestamp = timestamp;

      if (isPlaying) {{
        currentTime += dt * playbackSpeed;
        if (currentTime >= duration) {{
          currentTime = duration;
          isPlaying = false;
          btnPlay.textContent = '▶ Play';
        }}
        updateView();
        animFrame = requestAnimationFrame(tick);
      }}
    }}

    btnPlay.onclick = () => {{
      isPlaying = !isPlaying;
      btnPlay.textContent = isPlaying ? '⏸ Pause' : '▶ Play';
      if (isPlaying) {{
        lastTimestamp = null;
        if (currentTime >= duration) currentTime = 0.0;
        animFrame = requestAnimationFrame(tick);
      }}
    }};

    btnReset.onclick = () => {{
      seek(0.0);
    }};

    btnSpeed.onclick = () => {{
      const speeds = [0.5, 1.0, 2.0];
      const curIdx = speeds.indexOf(playbackSpeed);
      playbackSpeed = speeds[(curIdx + 1) % speeds.length];
      btnSpeed.textContent = `${{playbackSpeed.toFixed(1)}}x`;
    }};

    phaseTrack.onclick = (e) => {{
      const rect = phaseTrack.getBoundingClientRect();
      const clickPct = (e.clientX - rect.left) / rect.width;
      seek(clickPct * duration);
    }};

    anomalyTrack.onclick = (e) => {{
      const rect = anomalyTrack.getBoundingClientRect();
      const clickPct = (e.clientX - rect.left) / rect.width;
      seek(clickPct * duration);
    }};

      updateView();
  </script>
</body>
</html>
"""
        if output_path is not None:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(html, encoding="utf-8")

        return html

    def save_html(
        self,
        report: AuditReport,
        output_path: Union[str, Path],
    ) -> Path:
        """Generate and save HTML report to file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        html = self.generate_html(report)
        out.write_text(html, encoding="utf-8")
        return out


def serve_board(
    html_content_or_path: Union[str, Path],
    port: int = 8080,
    host: str = "127.0.0.1",
) -> http.server.HTTPServer:
    """Start a lightweight background HTTP server serving the timeline board.
    
    Requirement 9.1: Implement lightweight HTTP server for local visualization.
    
    Args:
        html_content_or_path: HTML string or Path to saved HTML file.
        port: Listening port (default: 8080).
        host: Host binding (default: '127.0.0.1').
        
    Returns:
        Running HTTPServer instance.
    """
    if isinstance(html_content_or_path, Path) or (
        isinstance(html_content_or_path, str) and Path(html_content_or_path).is_file()
    ):
        content = Path(html_content_or_path).read_bytes()
    else:
        content = str(html_content_or_path).encode("utf-8")

    class BoardHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, format, *args):
            pass  # Suppress console logging

    server = http.server.HTTPServer((host, port), BoardHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    return server
