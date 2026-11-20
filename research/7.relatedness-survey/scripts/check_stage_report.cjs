// Local report presentation checks only. No upstream research code or model runs.
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const {pathToFileURL, fileURLToPath} = require('url');
const base = path.resolve(__dirname, '..');
const owned = path.resolve(base, '../../.cache/relatedness-survey');
process.env.TMPDIR=path.join(owned,'report-tools/tmp');
process.env.XDG_CACHE_HOME=path.join(owned,'report-tools/cache');
fs.mkdirSync(process.env.TMPDIR,{recursive:true});
fs.mkdirSync(process.env.XDG_CACHE_HOME,{recursive:true});
process.env.LD_LIBRARY_PATH=path.join(owned,'report-tools/browser-libs/extracted/usr/lib/x86_64-linux-gnu');
process.env.FONTCONFIG_FILE=path.join(owned,'report-tools/report-fonts.conf');
fs.writeFileSync(process.env.FONTCONFIG_FILE,`<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "fonts.dtd"><fontconfig><dir>${path.join(owned,'report-tools/browser-libs/extracted/usr/share/fonts')}</dir><dir>/usr/share/fonts</dir><cachedir>${path.join(owned,'report-tools/cache/fontconfig')}</cachedir></fontconfig>`);
const {chromium} = require(path.join(owned, 'report-tools/playwright-core'));

(async () => {
  const target = path.join(base, 'report/0001-阶段工作汇报-20261120.html');
  const cache = path.join(owned, 'report-review/0001');
  fs.mkdirSync(cache, {recursive:true});
  const browser = await chromium.launch({
    executablePath:path.join(owned, 'report-tools/chromium_headless_shell-1194/chrome-linux/headless_shell'),
    headless:true,
    args:['--no-sandbox', '--disable-gpu']
  });
  const result = {checked_at:new Date().toISOString(), scope:'Stage report local layout/links/filter/content integrity only; no research/model execution', viewports:[], screenshots:[], errors:[]};
  for (const [name,width,height] of [['desktop',1440,1000],['mobile',390,844]]) {
    const page = await browser.newPage({viewport:{width,height}, deviceScaleFactor:1});
    await page.emulateMedia({reducedMotion:'reduce'});
    page.on('pageerror', e => result.errors.push(e.message));
    await page.goto(pathToFileURL(target).href);
    await page.evaluate(() => document.fonts.ready);
    const layout = await page.evaluate(() => ({
      document_width:document.documentElement.scrollWidth,
      viewport_width:innerWidth,
      section_count:document.querySelectorAll('main section').length,
      catalog_rows:document.querySelectorAll('#catalog-table tbody tr').length,
      table_wrappers:[...document.querySelectorAll('.table-wrap')].map(el=>({client:el.clientWidth,scroll:el.scrollWidth})),
      links:[...document.querySelectorAll('a[href]')].map(a=>a.getAttribute('href'))
    }));
    if(layout.document_width>width+1)throw new Error(name+' document overflow');
    if(layout.catalog_rows!==124 || layout.section_count!==13)throw new Error('Incomplete content');
    for(const destination of layout.links){
      if(/^(https?:|mailto:)/.test(destination))continue;
      const url=new URL(destination,pathToFileURL(target));
      if(destination.startsWith('#')){
        if(!await page.locator(destination).count())throw new Error('Missing anchor '+destination);
      } else if(!fs.existsSync(fileURLToPath(url))){throw new Error('Missing local link '+destination);}
    }
    result.viewports.push({name,width,height,document_width:layout.document_width,section_count:layout.section_count,catalog_rows:layout.catalog_rows,local_or_anchor_link_checks:layout.links.filter(x=>!/^https?:/.test(x)).length,table_overflow_confined:layout.table_wrappers});
    for(const [label,selector] of [['hero',null],['resource-overview','#section-6'],['acquisition-limits','#section-10']]){
      if(selector)await page.locator(selector).evaluate(el=>el.scrollIntoView({block:'start'}));
      else await page.evaluate(()=>scrollTo(0,0));
      if(selector && Math.abs((await page.locator(selector).boundingBox()).y)>32)throw new Error('Screenshot target not at viewport start: '+selector);
      const file=path.join(cache,`${name}-${label}.png`);
      await page.screenshot({path:file});
      const relative=path.relative(path.join(base,'report'),file);
      result.screenshots.push({path:relative,sha256:crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex')});
    }
    await page.locator('#catalog-search').fill('MIE-0014');
    const visible=await page.locator('#catalog-table tbody tr:visible').count();
    if(visible!==1)throw new Error('Catalog search does not isolate MIE-0014');
    await page.locator('#catalog-search').fill('');
    if(await page.locator('#catalog-table tbody tr:visible').count()!==124)throw new Error('Catalog reset failed');
    result.viewports.at(-1).filter_checked=true;
    await page.close();
  }
  await browser.close();
  if(result.errors.length)throw new Error(JSON.stringify(result.errors));
  const md=fs.readFileSync(path.join(base,'report/0001-阶段工作汇报-20261120.md'));
  const hash=crypto.createHash('sha256').update(md).digest('hex');
  if(!fs.readFileSync(target,'utf8').includes(hash))throw new Error('Markdown content hash mismatch');
  result.markdown_sha256=hash;
  result.html_sha256=crypto.createHash('sha256').update(fs.readFileSync(target)).digest('hex');
  result.human_visual_review='Pending actual screenshot inspection; mechanical checks are not visual approval';
  fs.writeFileSync(path.join(base,'report/0001-report-checks.json'),JSON.stringify(result,null,2)+'\n');
  console.log(JSON.stringify({viewports:result.viewports.map(x=>({name:x.name,width:x.width,document_width:x.document_width,sections:x.section_count,catalog_rows:x.catalog_rows,filter_checked:x.filter_checked})), errors:result.errors, screenshots:result.screenshots.length}));
})().catch(e=>{console.error(e);process.exitCode=1;});
