"use strict";
const el = id => document.getElementById(id);
const csrf = document.querySelector('meta[name="corpus-token"]').content;
let state, current, page, dirty=false, mode=null, start=null, boxes=[], busy=false;
let selectionVersion=0, draftRevision=null;
let recognitionJob=null, recognitionStarting=false;
const defects={"crossed-out":"划掉","occluded":"遮挡","overwritten":"重写重叠","erasure":"擦除残迹","faint":"过淡","blurred":"模糊","clipped":"裁切缺笔","mixed-writers":"混合书写者","printed-overlap":"印刷重叠","background-interference":"背景干扰","unknown":"其他待查"};
function status(text, error=false){
  el("status").textContent=text;
  const notice=el("actionStatus");
  notice.textContent=text;
  notice.hidden=false;
  notice.dataset.error=String(error);
}
async function api(path,body){const r=await fetch(path,body?{method:"POST",headers:{"Content-Type":"application/json","X-Corpus-Token":csrf},body:JSON.stringify(body)}:{});const d=await r.json();if(!r.ok)throw Error(d.error);return d;}
function asset(path){return "/asset?path="+encodeURIComponent(path);}
function actor(){const v=el("actor").value.trim();if(!v)throw Error("请填写审核人代号");return v;}
function preview(){if(!el("type").value.startsWith("math")){el("mathPreview").textContent="英文转写不做自动改写";return;}if(window.katex){try{katex.render(el("text").value,el("mathPreview"),{throwOnError:true,trust:false,strict:"warn",maxExpand:1000,displayMode:true});}catch(e){el("mathPreview").textContent="公式预览错误："+e.message;}}else el("mathPreview").textContent="离线 KaTeX 未安装，不能确认公式结构预览";}
function changes(statusValue){return {candidate_text:el("text").value,content_type:el("type").value,region_role:el("role").value,quality_status:el("quality").value,defects:[...el("defects").querySelectorAll("input:checked")].map(n=>n.value),notes:el("notes").value,privacy_checked:el("privacy").checked,review_status:statusValue};}
function persistDraft(){
  if(!current||!dirty)return;
  localStorage.setItem("corpus-draft:"+current.sample_id,JSON.stringify({
    revision:draftRevision??current.revision,changes:changes("needs-review")
  }));
}
function validateReview(value){
  const missing=[];
  if(!el("actor").value.trim())missing.push(["actor","填写页面顶部的审核人代号"]);
  if(value==="verified"){
    if(!el("text").value.trim())missing.push(["text","填写实际转写"]);
    if(el("type").value==="unknown")missing.push(["type","选择内容类型"]);
    if(["unknown","mixed"].includes(el("role").value))missing.push(["role","确认书写者；混合区域请先拆分"]);
    if(!el("attest").checked)missing.push(["attest","勾选对照原图的核验声明"]);
  }
  for(const id of ["actor","text","type","role","attest"])el(id).removeAttribute("aria-invalid");
  for(const [id] of missing)el(id).setAttribute("aria-invalid","true");
  if(missing.length)throw Error("尚未提交："+missing.map(item=>item[1]).join("；")+"。草稿已保留，可继续切换区域。");
  if(draftRevision!==null&&draftRevision!==current.revision){
    throw Error("这份草稿的修订已过期，原文仍保留；请先核对操作历史再处理冲突，可继续切换区域。");
  }
}
async function save(value="needs-review"){
  if(!current||busy)return;
  persistDraft();
  validateReview(value);
  const selected=current, before=visible().map(s=>s.sample_id);
  const index=before.indexOf(selected.sample_id);
  const request={sample_id:selected.sample_id,revision:selected.revision,actor:actor(),changes:changes(value),
    attestation:value==="verified"?{actual_transcription:true,crop_complete:true,writer_checked:true,human_review:true}:null};
  const controls=[...document.querySelectorAll("button,input,textarea,select")];
  const disabled=controls.map(node=>node.disabled);
  busy=true;
  controls.forEach(node=>node.disabled=true);
  status("正在保存审核记录…");
  let row;
  try{
    row=await api("/api/review",request);
    Object.assign(selected,row);
    dirty=false;
    draftRevision=null;
    localStorage.removeItem("corpus-draft:"+selected.sample_id);
  }finally{
    busy=false;
    controls.forEach((node,i)=>node.disabled=disabled[i]);
  }
  renderRegions();
  if(value==="verified"||value==="rejected"){
    const remaining=visible().filter(s=>s.sample_id!==selected.sample_id);
    const next=before.slice(index+1).map(id=>remaining.find(s=>s.sample_id===id)).find(Boolean)||remaining[0];
    if(next)await select(next.sample_id);
    else await select(null);
    status((value==="verified"?"已确认转写":"已排除区域")+" · 修订 "+row.revision+
      (next?"；已切换下一条。":"；本页当前队列已处理完，可切换队列查看记录。"));
  }else{
    el("sampleTitle").textContent=selected.sample_id+" · "+row.review_status;
    status("已保存待复核 · 修订 "+row.revision);
  }
}
function draft(){
  if(!current)return;
  dirty=true;
  el("attest").checked=false;
  persistDraft();
  status("草稿已保存在本机浏览器；可切换区域。保存或确认后才写入审核记录。");
  preview();
}
function useCandidate(r){
  if(!current||busy)return;
  el("text").value=r.text;
  if(r.engine==="unimernet"&&!el("type").value.startsWith("math"))el("type").value="math-expression";
  draft();
  status("已将候选填入草稿；请对照原图检查负号、上下标和大小写，再确认。");
}
function renderCandidates(){
  el("candidates").replaceChildren();
  if(!current)return;
  const rank=r=>r.candidate_method==="sample-2"?2:r.candidate_method==="sample-1"?1:0;
  const rows=state.recognition.filter(r=>r.sample_id===current.sample_id&&r.text.trim())
    .sort((a,b)=>Number(b.engine==="unimernet")-Number(a.engine==="unimernet")||rank(a)-rank(b));
  const seen=new Set();
  const unique=rows.filter(r=>{const key=r.engine+":"+r.text.trim();if(seen.has(key))return false;seen.add(key);return true;});
  if(!unique.length){
    const empty=document.createElement("p");empty.textContent="尚无识别候选。点“识别当前区域”生成 LaTeX；混合多行请先框出完整单行公式。";el("candidates").append(empty);
  }
  for(const [i,r] of unique.entries()){
    const card=document.createElement("article");card.className="candidate-card";
    const title=document.createElement("p");title.textContent=`候选 ${i+1} · ${r.engine}（待核验）`;
    const rendered=document.createElement("div");rendered.className="candidate-preview";
    if(r.engine==="unimernet"&&window.katex){
      try{katex.render(r.text,rendered,{throwOnError:true,trust:false,strict:"warn",maxExpand:1000,displayMode:true});}
      catch(e){rendered.textContent="预览失败，此候选可能不完整："+r.text;}
    }else rendered.textContent=r.text;
    const details=document.createElement("details"),summary=document.createElement("summary"),raw=document.createElement("pre");
    summary.textContent="查看转写源码与来源";raw.textContent=r.text+"\n来源运行："+(r.run||"未记录")+"\n解码："+(r.candidate_method||"原始识别")+(r.raw?.possible_token_truncation?"\n注意：可能达到识别长度上限":"");details.append(summary,raw);
    const choose=document.createElement("button");choose.textContent="使用候选 "+(i+1);choose.onclick=()=>useCandidate(r);
    card.append(title,rendered,choose,details);el("candidates").append(card);
  }
  const unavailable=!state.formula_recognition_available;
  el("recognizeRegion").disabled=unavailable||!!recognitionJob||recognitionStarting;
  el("recognizePage").disabled=unavailable||!!recognitionJob||recognitionStarting;
}
async function refreshCandidates(){
  const data=await api("/api/state");
  state.recognition=data.recognition;
  state.formula_recognition_available=data.formula_recognition_available;
  renderCandidates();
}
async function pollRecognition(){
  if(!recognitionJob)return;
  try{
    const job=await api("/api/recognize-status?id="+encodeURIComponent(recognitionJob));
    if(job.status==="running"){
      el("recognitionStatus").textContent=`正在本机识别 ${job.count} 个区域，每个最多3次解码；可以继续审核其他区域。`;
      setTimeout(pollRecognition,1500);return;
    }
    recognitionJob=null;localStorage.removeItem("corpus-recognition-job");
    await refreshCandidates();
    el("recognitionStatus").textContent=job.status==="complete"?"识别完成，候选已刷新；当前转写和草稿保持原样。":"识别失败："+job.error+"。已有候选仍可使用。";
  }catch(e){
    recognitionJob=null;localStorage.removeItem("corpus-recognition-job");renderCandidates();
    el("recognitionStatus").textContent="无法获取识别状态："+e.message+"。请点刷新候选查看已完成结果。";
  }
}
async function requestRecognition(scope){
  if(!current||recognitionJob||recognitionStarting)return;
  persistDraft();recognitionStarting=true;renderCandidates();
  try{
    const job=await api("/api/recognize",scope==="page"?{page_id:page.page_id}:{sample_id:current.sample_id});
    if(job.id){recognitionJob=job.id;localStorage.setItem("corpus-recognition-job",job.id);}
    else{await refreshCandidates();el("recognitionStatus").textContent=job.message;}
  }finally{recognitionStarting=false;renderCandidates();}
  if(recognitionJob)await pollRecognition();
}
function insertMath(before,after=""){
  if(!current||busy)return;
  const field=el("text"),start=field.selectionStart??field.value.length,end=field.selectionEnd??start;
  const selected=field.value.slice(start,end),value=before+selected+after;
  field.value=field.value.slice(0,start)+value+field.value.slice(end);
  if(!el("type").value.startsWith("math"))el("type").value="math-expression";
  field.focus();field.setSelectionRange(start+before.length,start+before.length+selected.length);draft();
}
async function select(id){
  if(busy)return;
  persistDraft();
  const version=++selectionVersion;
  current=state.samples.find(s=>s.sample_id===id);
  dirty=false;
  draftRevision=null;
  el("editing").hidden=!current;
  if(!current){drawOverlay();return;}
  mode=null;boxes=[];el("split").textContent="拆分当前区域";
  localStorage.setItem("corpus-selection",JSON.stringify({page:current.page_id,id}));
  let c=current;
  const draftText=localStorage.getItem("corpus-draft:"+id);
  if(draftText){
    const d=JSON.parse(draftText);
    c={...c,...d.changes};dirty=true;draftRevision=d.revision;
    status(d.revision===current.revision?"已恢复此区域的未提交草稿。":
      "已恢复旧草稿，但服务端已有新修订；请核对历史，暂不覆盖。",d.revision!==current.revision);
  }else status("已切换区域；对照原图核验后再确认。");
el("sampleTitle").textContent=id+" · "+current.review_status;el("crop").src=asset(c.image_crop);el("text").value=c.candidate_text;el("notes").value=c.notes;el("type").value=c.content_type;el("role").value=c.region_role;el("quality").value=c.quality_status||"unassessed";el("privacy").checked=c.privacy_checked;el("attest").checked=false;for(const input of el("defects").querySelectorAll("input"))input.checked=(c.defects||[]).includes(input.value);
el("risk").textContent=(c.quality_triage?.flags||[]).join("\n")||"尚无自动风险提示；这不代表区域清晰。请检查划改、笔画遮挡及其他书写者。";
if(c.review_status==="model-accepted")el("risk").textContent="此字符由 Sol 清洗并按你的授权采纳，未逐条人工确认。可查看或纠正，无需重复确认。\n"+el("risk").textContent;
renderCandidates();
preview();drawOverlay();
el("history").textContent="正在读取操作历史…";
const history=await api("/api/history?id="+id);
if(version===selectionVersion)el("history").textContent=JSON.stringify({source_hash:c.source_hash,page_id:c.page_id,bbox:c.bounding_box,writer:c.writer_provenance,history},null,2);
}
function visible(){return state.samples.filter(s=>s.page_id===page.page_id).sort((a,b)=>a.bounding_box[1]-b.bounding_box[1]||a.bounding_box[0]-b.bounding_box[0]).filter(s=>{const f=el("filter").value;if(f==="all")return true;if(f==="verified")return s.review_status==="verified";if(f==="model-accepted")return s.review_status==="model-accepted";if(["rejected","verified","model-accepted"].includes(s.review_status))return false;if(f==="symbols")return ["glyph","math-symbol"].includes(s.content_type);return f!=="risky"||(s.quality_triage?.flags?.length||s.confidence===null||s.quality_status!=="clean");});}
function renderRegions(){el("regions").replaceChildren();for(const [i,s] of visible().entries()){const div=document.createElement("div");div.className="region-choice";const cb=document.createElement("input");cb.type="checkbox";cb.value=s.sample_id;const b=document.createElement("button");b.textContent=(i+1)+" · "+s.content_type+" · "+s.review_status;b.onclick=()=>guard(()=>select(s.sample_id));div.append(cb,b);el("regions").append(div);}el("summary").textContent=`全库 ${state.pages.length} 页 / ${state.samples.length} 区域；本页 ${visible().length} 项。已核验不等于已满足生成训练质量。`;drawOverlay();}
function rect(box,cls,id){const r=document.createElementNS("http://www.w3.org/2000/svg","rect");r.setAttribute("x",box[0]);r.setAttribute("y",box[1]);r.setAttribute("width",box[2]-box[0]);r.setAttribute("height",box[3]-box[1]);r.setAttribute("class",cls);if(id)r.onclick=()=>{if(!mode)guard(()=>select(id));};el("overlay").append(r);}
function drawOverlay(){if(!page)return;el("overlay").replaceChildren();el("overlay").setAttribute("viewBox",`0 0 ${page.width} ${page.height}`);for(const s of visible())rect(s.bounding_box,"region"+(s.sample_id===current?.sample_id?" selected":""),s.sample_id);for(const b of boxes)rect(b,"drawn");}
async function load(){persistDraft();const oldPage=el("page").value||localStorage.getItem("corpus-page");state=await api("/api/state");el("page").replaceChildren();const docs=new Map(state.documents.map(d=>[d.document_id,d]));for(const p of [...state.pages].sort((a,b)=>docs.get(a.document_id).original_name.localeCompare(docs.get(b.document_id).original_name)||a.page_number-b.page_number)){const opt=document.createElement("option");opt.value=p.page_id;opt.textContent=docs.get(p.document_id).original_name+" · 第 "+p.page_number+" 页";el("page").append(opt);}if(state.pages.some(p=>p.page_id===oldPage))el("page").value=oldPage;await changePage();}
async function changePage(){
  if(busy)return;
  persistDraft();page=state.pages.find(p=>p.page_id===el("page").value);current=null;dirty=false;mode=null;boxes=[];el("editing").hidden=true;
  if(!page){status("尚无导入页面");return;}
  localStorage.setItem("corpus-page",page.page_id);el("original").src=asset(page.color_image);renderRegions();
  const bookmark=JSON.parse(localStorage.getItem("corpus-selection")||"null");
  const rows=visible();
  const chosen=rows.find(s=>s.sample_id===bookmark?.id)||rows.find(s=>localStorage.getItem("corpus-draft:"+s.sample_id))||rows[0];
  if(chosen)await select(chosen.sample_id);
  else status("当前队列没有区域，请切换队列或页面。");
}
async function guard(fn){if(busy)return;try{await fn();}catch(e){status("未完成："+e.message,true);}}
for(const [key,label] of Object.entries(defects)){const l=document.createElement("label");const input=document.createElement("input");input.type="checkbox";input.value=key;l.append(input,document.createTextNode(label));el("defects").append(l);}
el("actor").value=localStorage.getItem("corpus-actor")||"";el("actor").oninput=()=>localStorage.setItem("corpus-actor",el("actor").value);
for(const id of ["text","notes","type","role","quality","privacy","defects"])el(id).addEventListener("input",draft);
el("save").onclick=()=>guard(()=>save());el("verify").onclick=()=>guard(()=>save("verified"));el("reject").onclick=()=>guard(()=>save("rejected"));el("reload").onclick=()=>guard(load);el("page").onchange=()=>guard(changePage);el("filter").onchange=()=>guard(changePage);
el("recognizeRegion").onclick=()=>guard(()=>requestRecognition("region"));el("recognizePage").onclick=()=>guard(()=>requestRecognition("page"));el("refreshCandidates").onclick=()=>guard(async()=>{await refreshCandidates();el("recognitionStatus").textContent="候选已刷新，当前转写和草稿保持原样。";});
for(const [label,before,after] of [["分数","\\frac{","}{b}"],["根号","\\sqrt{","}"],["n次根号","\\sqrt[n]{","}"],["上标","^{","}"],["下标","_{","}"],["积分","\\int_{a}^{b} ","\\,dx"],["求和","\\sum_{i=1}^{n} ",""],["箭头 →","\\to ",""],["偏导 ∂","\\partial ",""],["矩阵","\\begin{pmatrix}a & b \\\\ c & d\\end{pmatrix}",""]]){
  const b=document.createElement("button");b.textContent=label;b.onclick=()=>insertMath(before,after);el("mathTools").append(b);
}
el("draw").onclick=()=>{mode="new";boxes=[];status("在原页拖动框选新区域");};
el("split").onclick=()=>guard(async()=>{if(!current)throw Error("请先选择父区域");if(mode!=="split"){if(dirty)await save();mode="split";boxes=[];el("split").textContent="完成拆分";status("在父区域内画至少两个框");}else{await api("/api/reshape",{items:[{id:current.sample_id,revision:current.revision}],boxes,actor:actor()});el("split").textContent="拆分当前区域";await load();}});
el("merge").onclick=()=>guard(async()=>{if(dirty)await save();const ids=[...el("regions").querySelectorAll("input:checked")].map(n=>n.value);await api("/api/reshape",{items:ids.map(id=>({id,revision:state.samples.find(s=>s.sample_id===id).revision})),actor:actor()});await load();});
function point(e){const r=el("overlay").getBoundingClientRect();return [Math.round((e.clientX-r.left)/r.width*page.width),Math.round((e.clientY-r.top)/r.height*page.height)];}
el("overlay").onpointerdown=e=>{if(!mode)return;start=point(e);el("overlay").setPointerCapture(e.pointerId);};
el("overlay").onpointerup=e=>guard(async()=>{if(!mode||!start)return;const end=point(e),box=[Math.min(start[0],end[0]),Math.min(start[1],end[1]),Math.max(start[0],end[0]),Math.max(start[1],end[1])];start=null;if(mode==="new"){const row=await api("/api/propose",{page_id:page.page_id,box,actor:actor()});await load();await select(row.sample_id);}else{boxes.push(box);drawOverlay();status("已画 "+boxes.length+" 个子区域；点完成拆分提交");}});
el("export").onclick=()=>guard(async()=>{if(dirty)await save();const r=await api("/api/export",{actor:actor()});status(`已导出 ${r.eligible_samples} 条合格样本：${r.path}`);});
window.addEventListener("beforeunload",e=>{if(dirty){e.preventDefault();e.returnValue="";}});
guard(async()=>{await load();recognitionJob=localStorage.getItem("corpus-recognition-job");if(recognitionJob){renderCandidates();await pollRecognition();}});
