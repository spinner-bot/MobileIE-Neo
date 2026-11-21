// Ordinary public publisher UI only: no account, captcha solving, stealth or control bypass.
// Website JavaScript may run; research source/models never run. Browser GPU disabled.
const fs=require('fs'),path=require('path'),crypto=require('crypto');
const owned='/data/Users/zrh/data/MobileIE-Neo/cache/relatedness-survey';
const base=path.resolve('research/7.relatedness-survey');
process.env.TMPDIR=path.join(owned,'report-tools/tmp');
process.env.XDG_CACHE_HOME=path.join(owned,'report-tools/cache');
process.env.LD_LIBRARY_PATH=path.join(owned,'report-tools/browser-libs/extracted/usr/lib/x86_64-linux-gnu');
process.env.FONTCONFIG_FILE=path.join(owned,'report-tools/report-fonts.conf');
const {chromium}=require(path.join(owned,'report-tools/playwright-core'));
const safeurl=u=>{try{let x=new URL(u);return x.origin+x.pathname}catch{return u}};
const record=(name,raw)=>{let f=path.join(owned,'archive',name);fs.writeFileSync(f,raw);return {archive:'archive/'+name,bytes:raw.length,sha256:crypto.createHash('sha256').update(raw).digest('hex')}};
(async()=>{
 const browser=await chromium.launch({executablePath:path.join(owned,'report-tools/chromium_headless_shell-1194/chrome-linux/headless_shell'),headless:true,args:['--no-sandbox','--disable-gpu']});
 const context=await browser.newContext({viewport:{width:1280,height:900},acceptDownloads:true});
 const page=await context.newPage();page.setDefaultTimeout(6000);
 const out={at:new Date().toISOString(),scope:'Ordinary publisher browser navigation; no login/captcha solving/stealth/researchsource/models/GPU',steps:[],responses:[]},pending=[];
 context.on('response',r=>{if(/sciencedirect\.com|elsevier\.com/.test(r.url()))out.responses.push({url:safeurl(r.url()),status:r.status(),content_type:r.headers()['content-type']})});
 page.on('download',d=>pending.push((async()=>{
  const target=path.join(owned,'archive','mie0027-browser-publisher001.pdf');await d.saveAs(target);const raw=fs.readFileSync(target);return {archive:'archive/'+path.basename(target),bytes:raw.length,sha256:crypto.createHash('sha256').update(raw).digest('hex'),PDF_magic:raw.subarray(0,4).toString()==='%PDF',error:await d.failure()};
 })()));
 try{
  const res=await page.goto('https://www.sciencedirect.com/science/article/pii/S2667379726000975',{waitUntil:'domcontentloaded',timeout:25000});
  await page.waitForTimeout(1500);let text=await page.locator('body').innerText();
  out.steps.push({event:'loaded',url:safeurl(page.url()),status:res?.status(),visible_text:text.slice(0,5000)});
  out.initial_HTML=record('mie0027-browser-publisher001.html',Buffer.from(await page.content()));
  out.initial_screenshot=record('mie0027-browser-publisher001.png',await page.screenshot());
  if(/captcha|verify you are human|access denied|unusual traffic|security check|verify.*browser/i.test(text)||res?.status()===403){out.stopped_for_access_challenge=true;}
  else{
   const pdf=page.locator('a[href*="pdfft"]').first();out.pdf_link_count=await pdf.count();
   if(out.pdf_link_count){await pdf.click({timeout:8000});out.normal_pdf_link_clicked=true;await page.waitForTimeout(6000);}
   out.final_pages=context.pages().map(p=>safeurl(p.url()));
  }
 }catch(e){out.error=String(e);out.last_url=safeurl(page.url());}
 out.downloads=await Promise.allSettled(pending);
 await browser.close();
 fs.writeFileSync(path.join(base,'resources/MIE-0027/derived/docs/publisher-browser-attempt001.json'),JSON.stringify(out,null,2)+'\n');
 console.log(JSON.stringify({steps:out.steps.map(x=>({event:x.event,status:x.status,url:x.url})),stopped_for_access_challenge:out.stopped_for_access_challenge,error:out.error,downloads:out.downloads}));
})().catch(e=>{console.error(String(e));process.exitCode=1});
