---
# the default layout is 'page'
icon: fas fa-info-circle
order: 4
---

<style>
.kd-hero {
  display: flex;
  align-items: center;
  gap: 1.4rem;
  margin: 0.4rem 0 1.4rem;
}
.kd-avatar {
  flex: 0 0 auto;
  width: 86px;
  height: 86px;
  border-radius: 50%;
  object-fit: cover;
  box-shadow: 0 0 0 4px rgba(128, 128, 128, 0.15);
}
.kd-name {
  font-size: 1.55rem;
  font-weight: 700;
  margin: 0 0 0.15rem;
}
.kd-tag {
  margin: 0 0 0.35rem;
  font-weight: 500;
}
.kd-loc {
  margin: 0;
  font-size: 0.88rem;
  opacity: 0.75;
}
.kd-dot {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #3bd80d;
  margin-right: 0.3rem;
  vertical-align: middle;
}
.kd-contact {
  display: flex;
  flex-wrap: wrap;
  gap: 0.7rem;
  margin: 1.4rem 0 1.8rem;
}
.kd-contact a {
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  padding: 0.4rem 0.9rem;
  border: 1px solid var(--btn-border-color, #c7c7c7);
  border-radius: 0.7rem;
  text-decoration: none;
}
.kd-contact a:hover {
  border-color: var(--btn-border-hover-color, #a0a0a0);
}
.kd-skills {
  display: grid;
  gap: 0.5rem;
  margin: 1rem 0 1.6rem;
}
.kd-skill-row {
  display: grid;
  grid-template-columns: 130px 1fr;
  gap: 0.9rem;
  padding-bottom: 0.5rem;
  border-bottom: 1px dashed rgba(128, 128, 128, 0.28);
  font-size: 0.95rem;
}
.kd-skill-row:last-child {
  border-bottom: none;
  padding-bottom: 0;
}
.kd-skill-label {
  font-weight: 600;
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  opacity: 0.55;
  padding-top: 0.15rem;
}
.kd-focus {
  display: inline-block;
  padding: 0.22rem 0.8rem;
  margin: 0.2rem 0.25rem 0.2rem 0;
  border: 1px solid var(--main, #2a408e);
  color: var(--main, #2a408e);
  border-radius: 1rem;
  font-size: 0.87rem;
  font-weight: 500;
}
.kd-timeline {
  list-style: none;
  margin: 0;
  padding: 0;
  position: relative;
}
.kd-timeline::before {
  content: "";
  position: absolute;
  left: 8px;
  top: 6px;
  bottom: 6px;
  width: 2px;
  background: rgba(128, 128, 128, 0.35);
}
.kd-timeline > li {
  position: relative;
  padding: 0 0 1.8rem 2.2rem;
  margin: 0;
}
.kd-timeline > li:last-child {
  padding-bottom: 0.3rem;
}
.kd-timeline > li::before {
  content: "";
  position: absolute;
  left: 0;
  top: 5px;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: var(--main, #2a408e);
  box-shadow: 0 0 0 3px rgba(128, 128, 128, 0.18);
}
.kd-when {
  font-size: 0.86em;
  opacity: 0.75;
  margin-bottom: 0.15rem;
}
.kd-role {
  font-weight: 600;
  font-size: 1.05em;
  margin-bottom: 0.4rem;
}
.kd-timeline ul {
  margin: 0;
  padding-left: 1.2rem;
}
.kd-timeline ul li {
  margin: 0.25rem 0;
}
@media (max-width: 480px) {
  .kd-hero {
    flex-direction: column;
    align-items: flex-start;
  }
}
</style>

<div class="kd-hero">
  <div>
    <h2 class="kd-name">Kushal Dabbe</h2>
    <p class="kd-tag">LLM inference · ML systems · Computer vision</p>
    <p class="kd-loc"><span class="kd-dot"></span>Software Engineer at Rakuten Mobile · Tokyo, Japan</p>
  </div>
</div>

I build ML systems end-to-end. LLM agents, retrieval pipelines, and computer
vision. My focus now is the **LLM inference and training stack**: serving
optimization, distributed training, fine-tuning, and benchmarking. This site
tracks the projects and notes from that work.

## Experience

<ul class="kd-timeline">
  <li>
    <div class="kd-when">Jan 2024 - Present · Tokyo, Japan</div>
    <div class="kd-role">Software Engineer · Rakuten Mobile</div>
  </li>
  <li>
    <div class="kd-when">Aug 2023 - Dec 2023 · Remote (Delhi, India)</div>
    <div class="kd-role">NLP Engineer · Counsello AI</div>
  </li>
  <li>
    <div class="kd-when">2019 - 2023</div>
    <div class="kd-role">IIT (BHU) Varanasi</div>
    <div>B.Tech, Mechanical Engineering</div>
  </li>
</ul>

## Skills

<div class="kd-skills">
  <div class="kd-skill-row"><span class="kd-skill-label">Languages</span><span>Python · C++ · C · SQL · JavaScript · Go</span></div>
  <div class="kd-skill-row"><span class="kd-skill-label">AI / LLM</span><span>PyTorch · HuggingFace · RAG pipelines · LangChain · LangGraph · Pinecone/ChromaDB)</span></div>
  <div class="kd-skill-row"><span class="kd-skill-label">Vision</span><span>YOLOv8 fine-tuning · Grounding DINO · SAM 2 · Depth Anything V2 · BoT-SORT tracking · pose estimation · OpenCV</span></div>
  <div class="kd-skill-row"><span class="kd-skill-label">ML optimization</span><span>ONNX export · graph optimization · CoreML / CUDA deployment · mAP / IoU evaluation</span></div>
  <div class="kd-skill-row"><span class="kd-skill-label">Infra</span><span>FastAPI · Next.js · Spark · Airflow · AWS · Docker</span></div>
</div>

## Currently focused on

<div class="kd-skills">
  <div class="kd-skill-row"><span class="kd-skill-label">Training</span><span>DDP · FSDP · DeepSpeed · mixed precision</span></div>
  <div class="kd-skill-row"><span class="kd-skill-label">Fine-tuning</span><span>LoRA · QLoRA · SFT · DPO · dataset curation</span></div>
  <div class="kd-skill-row"><span class="kd-skill-label">Inference</span><span>vLLM · SGLang · TensorRT-LLM · quantization</span></div>
  <div class="kd-skill-row"><span class="kd-skill-label">Serving</span><span>Triton · Kubernetes</span></div>
  <div class="kd-skill-row"><span class="kd-skill-label">Evals</span><span>lm-eval-harness · benchmark design</span></div>
</div>

<div class="kd-contact">
  <a href="mailto:kushaldabbe752@gmail.com"><i class="fas fa-envelope"></i> Email</a>
  <a href="https://www.linkedin.com/in/kushal-dabbe-5b28441a0/"><i class="fab fa-linkedin"></i> LinkedIn</a>
  <a href="https://github.com/kushaldabbe"><i class="fab fa-github"></i> GitHub</a>
</div>
