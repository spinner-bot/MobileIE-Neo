// Render static author SVG assets for inspection. Original bytes unchanged.
// Website/network/model code is not run; browser JS and GPU disabled.
const fs=require('fs'),path=require('path'),crypto=require('crypto');
const root=path.resolve('research/7.relatedness-survey'),res=path.join(root,'resources/MIE-0023');
const owned='/data/Users/zrh/data/MobileIE-Neo/cache/relatedness-survey';
process.env.TMPDIR=path.join(owned,'report-tools/tmp');process.env.XDG_CACHE_HOME=path.join(owned,'report-tools/cache');
process.env.LD_LIBRARY_PATH=path.join(owned,'report-tools/browser-libs/extracted/usr/lib/x86_64-linux-gnu');
process.env.FONTCONFIG_FILE=path.join(owned,'report-tools/report-fonts.conf');
const {chromium}=require(path.join(owned,'report-tools/playwright-core'));
(async()=>{
 const browser=await chromium.launch({executablePath:path.join(owned,'report-tools/chromium_headless_shell-1194/chrome-linux/headless_shell'),headless:true,args:['--no-sandbox','--disable-gpu']});
 const ctx=await browser.newContext({javaScriptEnabled:false,offline:true});const assets=[];
 const targetdir=path.join(owned,'derived/MIE-0023/media');fs.mkdirSync(targetdir,{recursive:true});
 for(const name of ['benchmark_arch','fjnd_modeling','training_pipeline']){
  const source=path.join(res,'derived/docs/artifacts/assets',name+'.svg');const raw=fs.readFileSync(source);let svg=raw.toString();const w=Number(svg.match(/<svg[^>]*\bwidth="(\d+)"/)[1]),h=Number(svg.match(/<svg[^>]*\bheight="(\d+)"/)[1]);
  svg=svg.replace(/<svg\b/,`<svg viewBox="0 0 ${w} ${h}"`).replace(/(<svg[^>]*\bwidth=")\d+"/,'$11400"').replace(/(<svg[^>]*\bheight=")\d+"/,`$1${Math.ceil(h*1400/w)}"`);
  const page=await ctx.newPage({viewport:{width:1420,height:Math.ceil(h*1400/w)+20}});await page.setContent('<html><body style="margin:0;background:white">'+svg+'</body></html>');
  const dest=path.join(targetdir,name+'.png');await page.locator('svg').screenshot({path:dest,timeout:10000});await page.close();
  const local=path.join(res,'derived/figures',name+'.png');fs.mkdirSync(path.dirname(local),{recursive:true});fs.symlinkSync(path.relative(path.dirname(local),dest),local);
  assets.push({source:'derived/docs/artifacts/assets/'+name+'.svg',source_sha256:crypto.createHash('sha256').update(raw).digest('hex'),path:'derived/figures/'+name+'.png',sha256:crypto.createHash('sha256').update(fs.readFileSync(dest)).digest('hex'),scope:'Static SVG fit-to-width derivative for viewing; originalunchanged, not itself reading claim'});
 }
 await browser.close();fs.writeFileSync(path.join(res,'derived/docs/media-assets.json'),JSON.stringify({at:new Date().toISOString(),assets,method:'OwnedChromium JSdisabled/offline/GPUdisabled, addviewBox/resize onlyinmemory forstaticrender'},null,2)+'\n');console.log('Rendered3staticSVGs; no reading approval/model execution');
})();
