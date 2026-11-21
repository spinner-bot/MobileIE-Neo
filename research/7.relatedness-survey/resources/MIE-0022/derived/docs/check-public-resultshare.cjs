// Normal public author-share navigation only; no account login or captcha bypass.
// Runs website UI, never research model/source code. Browser GPU is disabled.
const fs=require('fs'),path=require('path'),crypto=require('crypto');
const base=path.resolve('research/7.relatedness-survey');
const owned='/data/Users/zrh/data/MobileIE-Neo/cache/relatedness-survey';
process.env.TMPDIR=path.join(owned,'report-tools/tmp');
process.env.XDG_CACHE_HOME=path.join(owned,'report-tools/cache');
process.env.LD_LIBRARY_PATH=path.join(owned,'report-tools/browser-libs/extracted/usr/lib/x86_64-linux-gnu');
process.env.FONTCONFIG_FILE=path.join(owned,'report-tools/report-fonts.conf');
const {chromium}=require(path.join(owned,'report-tools/playwright-core'));
(async()=>{
 const browser=await chromium.launch({executablePath:path.join(owned,'report-tools/chromium_headless_shell-1194/chrome-linux/headless_shell'),headless:true,args:['--no-sandbox','--disable-gpu']});
 const page=await browser.newPage({viewport:{width:1280,height:900}});
 page.setDefaultTimeout(5000);
 const requests=[],responseTasks=[];page.context().on('request',r=>{if(/download|baidupcs|pcs\.baidu/i.test(r.url()))requests.push({url:r.url(),method:r.method()});});page.context().on('response',r=>{if(/sharedownload|share\/download/i.test(r.url()))responseTasks.push((async()=>({url:r.url(),status:r.status(),body:await r.text().catch(e=>String(e))}))());});
 const downloads=[];page.on('download',d=>{downloads.push((async()=>{const safe=path.basename(d.suggestedFilename()).replace(/[^A-Za-z0-9._-]/g,'_');const target=path.join(owned,'archive','mie0022-author-'+safe);await d.saveAs(target);const raw=fs.readFileSync(target);return {file:path.basename(target),bytes:raw.length,sha256:crypto.createHash('sha256').update(raw).digest('hex'),error:await d.failure()};})());});
 const out={at:new Date().toISOString(),scope:'Normal public-share UI; no login, captcha solving, research source or GPU execution',steps:[]};
 try{
  await page.goto('https://pan.baidu.com/s/1itaDTTeSkgUtnzcZWefCNA?pwd=1224',{waitUntil:'commit',timeout:20000});
  await page.waitForTimeout(1500);
  await page.locator('#accessCode').fill('1224');
  await page.getByText('提取文件',{exact:true}).click();
  await page.getByText('下载',{exact:true}).first().waitFor({timeout:12000});
  await page.getByText('下载',{exact:true}).first().click();
  await page.waitForTimeout(1800);
  
  for(let poll=0;poll<8 && downloads.length===0;poll++){const status=await page.locator('body').innerText();if(!status.includes('正在获取下载链接') && poll>0)break;await page.waitForTimeout(1800);}
  if(await page.getByText('普通下载',{exact:true}).count()){await page.getByText('普通下载',{exact:true}).first().click();out.normal_download_clicked=true;for(let p=0;p<8 && downloads.length===0;p++){await page.waitForTimeout(1800);}}
    out.steps.push({event:'loaded',url:page.url(),body:(await page.locator('body').innerText()).slice(0,2500)});
  out.inputs=await page.locator('input').evaluateAll(a=>a.map(x=>({type:x.type,name:x.name,id:x.id,placeholder:x.placeholder}))); 
  out.buttons=await page.locator('button,a').allTextContents();

 }catch(e){out.error=String(e);out.partial_url=page.url();try{out.partial_body=(await page.locator('body').innerText({timeout:2000})).slice(0,2000);}catch(e2){out.partial_capture_error=String(e2)}}
 out.pages=page.context().pages().map(p=>{try{const u=new URL(p.url());return u.origin+u.pathname}catch(e){return p.url()}});
 out.downloads=await Promise.all(downloads);
 const network={requests,responses:await Promise.all(responseTasks)};const netraw=Buffer.from(JSON.stringify(network,null,2));const netfile=path.join(owned,'archive','mie0022-resultshare-network007.json');fs.writeFileSync(netfile,netraw);out.network_trace={archive:'archive/'+path.basename(netfile),bytes:netraw.length,sha256:crypto.createHash('sha256').update(netraw).digest('hex'),requests:requests.map(r=>{const u=new URL(r.url);return{method:r.method,path:u.origin+u.pathname}})};
 await browser.close();
 fs.writeFileSync(path.join(base,'resources/MIE-0022/derived/docs/resultshare-browser-attempt007.json'),JSON.stringify(out,null,2)+'\n');
 console.log(JSON.stringify(out));
})();
