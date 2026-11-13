/* Archive deinterlacer UI: thin client over the core engine REST API. */

const state = {
  jobId: null,
  rawCount: 0,
  resultCount: 0,
  outputFps: 0,
  currentRaw: 0,
  currentResult: 0,
  manifest: null,
  timer: null,
};

const $ = (id) => document.getElementById(id);

function setStatus(el, text, kind = "") {
  el.textContent = text;
  el.className = "status " + kind;
}

async function pollJob() {
  const res = await fetch(`/api/jobs/${state.jobId}`);
  const data = await res.json();
  if (!data.ok && data.status === "error") {
    setStatus($("upload-status"), "失败：" + data.error, "error");
    clearInterval(state.timer);
    return;
  }
  if (data.status === "analyzed") {
    clearInterval(state.timer);
    state.rawCount = data.raw_frame_count;
    renderMetadata(data);
    renderDiagnosis(data.diagnosis);
    $("meta-card").classList.remove("hidden");
    $("control-card").classList.remove("hidden");
    $("viewer-card").classList.remove("hidden");
    setStatus($("upload-status"), "诊断完成，请显式选择场序", "ok");
    showRawFrame(0);
  } else if (data.status === "processing") {
    setStatus($("process-status"), "内核处理中……", "busy");
  } else if (data.status === "done") {
    clearInterval(state.timer);
    state.resultCount = data.output_frame_count;
    state.outputFps = data.output_fps;
    state.manifest = data.manifest;
    renderExport(data);
    showResultFrame(0);
    setStatus(
      $("process-status"),
      `完成：${data.output_frame_count} 帧 @ ${data.output_fps} fps`,
      "ok"
    );
  }
}

function renderMetadata(data) {
  const rows = [
    ["宽度 × 高度", `${data.info.width} × ${data.info.height}`],
    ["容器帧率", `${data.info.fps} fps`],
    ["隔行帧数", data.info.frame_count],
    ["时长（容器）", `${data.info.duration.toFixed(3)} s`],
    ["容器声明场序", data.info.metadata_field_order],
  ];
  $("meta-table").innerHTML = rows
    .map(([k, v]) => `<tr><th>${k}</th><td>${v}</td></tr>`)
    .join("");
}

function renderDiagnosis(d) {
  const box = $("diagnosis-box");
  const blocked = d.blocking !== "none";
  box.innerHTML = `
    <p>内容判定：<strong>${d.content_verdict}</strong>
      <span class="badge ${blocked ? "block" : d.confidence}">${d.confidence} confidence</span>
      ${blocked ? '<span class="badge block">blocked: ' + d.blocking + "</span>" : ""}
    </p>
    <p>建议场序：<strong>${d.suggested_order}</strong>
       （TFF 时序平滑度 ${d.order_score_tff}，BFF ${d.order_score_bff}，越低越平滑）</p>
    <p>梳齿能量 ${d.comb_energy} ｜ 含梳齿帧比例 ${d.comb_frame_ratio}
       ｜ 逐行相似度 ${d.progressive_score}
       ｜ 偶数高度 ${d.even_height}</p>`;
  $("reasons").innerHTML = d.reasons.map((r) => `<li>${r}</li>`).join("");
}

function showRawFrame(i) {
  state.currentRaw = Math.max(0, Math.min(i, state.rawCount - 1));
  const order = $("field-order").value || "tff";
  $("raw-img").src =
    `/api/jobs/${state.jobId}/raw/${state.currentRaw}?t=${Date.now()}`;
  // Two fields per interlaced frame; show the earlier captured field.
  const fieldIndex = order === "bff"
    ? state.currentRaw * 2 + 1
    : state.currentRaw * 2;
  $("field-img").src =
    `/api/jobs/${state.jobId}/field/${fieldIndex}?order=${order}&t=${Date.now()}`;
  updateFrameLabel();
}

function showResultFrame(i) {
  if (!state.resultCount) return;
  state.currentResult = Math.max(0, Math.min(i, state.resultCount - 1));
  const overlay = $("motion-overlay").checked ? "&overlay=motion" : "";
  $("result-img").src =
    `/api/jobs/${state.jobId}/result/${state.currentResult}?t=${Date.now()}${overlay}`;
  updateFrameLabel();
  updateTiming();
}

function updateFrameLabel() {
  $("frame-label").textContent =
    `隔行帧 ${state.currentRaw + 1}/${state.rawCount}` +
    (state.resultCount
      ? `　重建帧 ${state.currentResult + 1}/${state.resultCount}`
      : "");
}

function updateTiming() {
  if (!state.manifest) {
    $("timing-box").textContent = "运行处理后显示场/帧时间戳";
    return;
  }
  const f = state.manifest.frames[state.currentResult];
  const fieldInfo = state.manifest.fields
    .filter((x) => f.source_field_indices.includes(x.field))
    .map((x) => `场#${x.field}（${x.position === "top" ? "偶数行/上场位置" : "奇数行/下场位置"}，t=${x.timestamp_seconds}s）`)
    .join("\n");
  $("timing-box").textContent =
    `输出帧率 ${state.manifest.output_fps} fps ｜ 帧周期 ${state.manifest.frame_period_seconds}s\n` +
    `重建帧#${f.frame} t=${f.timestamp_seconds}s 来源：\n${fieldInfo}\n` +
    `总帧数 ${state.manifest.output_frame_count}，总时长 ${state.manifest.expected_duration_seconds}s` +
    `（帧数/帧率一致性：${state.manifest.duration_check}）`;
}

function renderExport(data) {
  $("export-card").classList.remove("hidden");
  const m = data.manifest;
  $("export-info").innerHTML =
    `模式 <b>${m.mode}</b>，场序 <b>${m.field_order}</b><br>` +
    `输出 ${m.output_frame_count} 帧 @ ${m.output_fps} fps，` +
    `时长 ${m.expected_duration_seconds} s，运动像素占比 ${m.motion_pixel_ratio}<br>` +
    `文件以 progressive 标记编码，播放器不会再触发自带去交错。`;
  $("dl-video").href = `/api/jobs/${state.jobId}/download/video`;
  $("dl-manifest").href = `/api/jobs/${state.jobId}/download/manifest`;
}

$("upload-btn").addEventListener("click", async () => {
  const file = $("file").files[0];
  if (!file) {
    setStatus($("upload-status"), "请先选择视频文件", "error");
    return;
  }
  const fd = new FormData();
  fd.append("video", file);
  setStatus($("upload-status"), "上传中……", "busy");
  const res = await fetch("/api/upload", { method: "POST", body: fd });
  const data = await res.json();
  if (!data.ok) {
    setStatus($("upload-status"), "失败：" + data.error, "error");
    return;
  }
  state.jobId = data.job_id;
  setStatus($("upload-status"), "解码与诊断中……", "busy");
  clearInterval(state.timer);
  state.timer = setInterval(pollJob, 700);
});

$("process-btn").addEventListener("click", async () => {
  const order = $("field-order").value;
  if (!order) {
    setStatus(
      $("process-status"),
      "内核拒绝运行：场序必须由操作员显式选择，工具不会猜测",
      "error"
    );
    return;
  }
  const body = {
    field_order: order,
    mode: $("mode").value,
    thresholds: {
      diff_threshold: parseFloat($("diff-threshold").value),
      neighborhood_threshold: parseFloat($("neighborhood-threshold").value),
    },
  };
  const res = await fetch(`/api/jobs/${state.jobId}/process`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!data.ok) {
    setStatus($("process-status"), "内核拒绝：" + data.error, "error");
    return;
  }
  setStatus($("process-status"), "内核处理中……", "busy");
  clearInterval(state.timer);
  state.timer = setInterval(pollJob, 500);
});

$("field-order").addEventListener("change", () => showRawFrame(state.currentRaw));
$("prev-btn").addEventListener("click", () => {
  showRawFrame(state.currentRaw - 1);
  showResultFrame(state.currentResult - 2);
});
$("next-btn").addEventListener("click", () => {
  showRawFrame(state.currentRaw + 1);
  showResultFrame(state.currentResult + 2);
});
$("motion-overlay").addEventListener("change", () =>
  showResultFrame(state.currentResult)
);
