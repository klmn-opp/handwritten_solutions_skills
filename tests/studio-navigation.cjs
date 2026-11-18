// Deterministic front-end state tests; real browser/HTTP evidence is recorded separately.
const assert = require("node:assert/strict");
const { test } = require("node:test");
const fs = require("node:fs");
const vm = require("node:vm");
const source = fs.readFileSync("research/corpus/web/studio.js", "utf8");

class Element {
  constructor(tag="div") {
    Object.assign(this, {tag, children:[], value:"", textContent:"", checked:false,
      disabled:false, hidden:false, attributes:{}, dataset:{}});
  }
  append(...children) { this.children.push(...children); if(this.tag==="select"&&!this.value)this.value=children[0]?.value||""; }
  replaceChildren(...children) { this.children=[]; this.append(...children); }
  setAttribute(name,value) { this.attributes[name]=value; }
  removeAttribute(name) { delete this.attributes[name]; }
  addEventListener() {}
  querySelectorAll(selector) {
    const all=this.children.flatMap(c=>[c,...(c.querySelectorAll?.("*")||[])]);
    return all.filter(c=>selector==="*"||(c.tag==="input"&&(!selector.includes(":checked")||c.checked)));
  }
}

async function app(storage=new Map()) {
  const ids=[...source.matchAll(/el\("([A-Za-z]+)"\)/g)].map(m=>m[1]);
  const elements=Object.fromEntries(ids.map(id=>[id,new Element()]));
  for(const id of ["page","filter","type","role","quality"])elements[id].tag="select";
  elements.filter.value="pending";
  const database={documents:[{document_id:"d",original_name:"fixture.png"}],
    pages:[{page_id:"p",document_id:"d",page_number:1,width:400,height:240,color_image:"page.png"}],
    samples:["a","b","c"].map((id,i)=>({sample_id:id,page_id:"p",revision:0,review_status:"auto-candidate",
      bounding_box:[0,i*70,390,i*70+60],image_crop:id+".png",candidate_text:id,
      content_type:"english-line",region_role:"personal-handwriting",quality_status:"unassessed",
      notes:"",defects:[],privacy_checked:false,writer_provenance:{domain:"synthetic"}})), recognition:[]};
  const posts=[];
  let failure=null;
  const document={getElementById:id=>elements[id],querySelector:()=>({content:"test-token"}),
    createElement:tag=>new Element(tag),createElementNS:(_,tag)=>new Element(tag),
    createTextNode:text=>({textContent:text}),querySelectorAll:()=>Object.values(elements)};
  const context=vm.createContext({document,window:{addEventListener(){}},localStorage:{
    getItem:key=>storage.get(key)??null,setItem:(key,value)=>storage.set(key,value),removeItem:key=>storage.delete(key)},
    fetch:async(path,options)=>{
      let data;
      if(options?.method==="POST"){
        const body=JSON.parse(options.body);posts.push(body);
        if(failure)return {ok:false,json:async()=>({error:failure})};
        const row=database.samples.find(s=>s.sample_id===body.sample_id);
        Object.assign(row,body.changes,{revision:row.revision+1});data=row;
      }else data=path==="/api/state"?database:[];
      return {ok:true,json:async()=>structuredClone(data)};
    }});
  const run=code=>vm.runInContext(code,context);
  run(source);
  await new Promise(setImmediate);
  return {elements,storage,posts,database,run,fail:message=>{failure=message;}};
}

test("an incomplete review does not block changing regions; text and notes survive",async()=>{
  const a=await app();await a.run('select("b")');
  a.elements.text.value="corrected second region";a.elements.notes.value="uncertain sign";a.run("draft()");
  await a.run('select("c")');
  assert.equal(a.run("current.sample_id"),"c");assert.equal(a.posts.length,0);
  await a.run('select("b")');
  assert.equal(a.elements.text.value,"corrected second region");
  assert.equal(a.elements.notes.value,"uncertain sign");
  assert.equal(a.elements.attest.checked,false);
});

test("missing reviewer and attestation show beside the action and never submit",async()=>{
  const a=await app();a.elements.text.value="changed";a.run("draft()");
  await a.run('guard(()=>save("verified"))');
  assert.match(a.elements.actionStatus.textContent,/审核人/);
  assert.match(a.elements.actionStatus.textContent,/核验声明/);
  assert.equal(a.elements.actionStatus.hidden,false);assert.equal(a.posts.length,0);
  assert.ok(a.storage.has("corpus-draft:a"));
  await a.run('select("b")');assert.equal(a.run("current.sample_id"),"b");
});

test("successful confirmation updates the record, clears only that draft and advances",async()=>{
  const a=await app();await a.run('select("b")');
  a.elements.text.value="second correction";a.run("draft()");
  a.elements.actor.value="fixture reviewer";a.elements.attest.checked=true;
  await a.run('save("verified")');
  assert.equal(a.posts.length,1);assert.equal(a.posts[0].sample_id,"b");
  assert.equal(a.posts[0].attestation.human_review,true);
  assert.equal(a.database.samples[1].review_status,"verified");
  assert.equal(a.run("current.sample_id"),"c");assert.equal(a.storage.has("corpus-draft:b"),false);
  assert.match(a.elements.actionStatus.textContent,/已确认转写.*下一条/);
});

test("failed save releases controls and allows navigation without losing the draft",async()=>{
  const a=await app();a.elements.actor.value="fixture reviewer";
  a.elements.text.value="keep after network failure";a.run("draft()");a.fail("server unavailable");
  await a.run("guard(()=>save())");
  assert.equal(a.run("busy"),false);assert.equal(a.elements.verify.disabled,false);
  assert.match(a.elements.actionStatus.textContent,/server unavailable/);
  await a.run('select("c")');await a.run('select("a")');
  assert.equal(a.elements.text.value,"keep after network failure");assert.equal(a.posts.length,1);
});

test("reload restores last selected region and unsubmitted draft without an actor",async()=>{
  const first=await app();await first.run('select("b")');
  first.elements.text.value="resume second";first.run("draft()");
  const second=await app(first.storage);
  assert.equal(second.run("current.sample_id"),"b");
  assert.equal(second.elements.text.value,"resume second");assert.equal(second.posts.length,0);
});

test("pre-fix drafts with no selection bookmark are restored",async()=>{
  const storage=new Map([["corpus-page","p"],["corpus-draft:b",JSON.stringify({revision:0,changes:{candidate_text:"old draft"}})]]);
  const a=await app(storage);
  assert.equal(a.run("current.sample_id"),"b");assert.equal(a.elements.text.value,"old draft");
});

test("outdated drafts remain visible and cannot overwrite a newer revision",async()=>{
  const storage=new Map([["corpus-draft:b",JSON.stringify({revision:-1,changes:{candidate_text:"old pending text"}})]]);
  const a=await app(storage);a.elements.actor.value="fixture reviewer";
  await a.run("guard(()=>save())");assert.equal(a.posts.length,0);
  assert.match(a.elements.actionStatus.textContent,/修订已过期/);
  await a.run('select("c")');await a.run('select("b")');
  assert.equal(a.elements.text.value,"old pending text");assert.equal(JSON.parse(storage.get("corpus-draft:b")).revision,-1);
});
