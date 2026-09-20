"""Local-browser QA only. Requires Playwright; never visits the hosted Site."""

import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from aster.tokenizer.artifact import load_tokenizer

CORPUS = ROOT / 'runs/b22e1cfb780c4acdb698839356c9d52b/observation-bundle.json'
TOKENS = ROOT / 'runs/07032cd0a170411dad43117c9b779c81/tokenizer-bundle.json'
MODEL = ROOT / 'artifacts/tokenizers/0f84d39c8d7726383155e5a4f9e3dd7e0bb7782b0a3ba48b2ed1b0c28d4b160c'
OUT = ROOT / 'reports/generated/workbench'
OUT.mkdir(parents=True, exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, executable_path='/home/zhong/.cache/ms-playwright/chromium-1208/chrome-linux64/chrome', args=['--no-sandbox'])
    page = browser.new_page(viewport={'width': 1512, 'height': 1050}, device_scale_factor=1)
    errors, uploads = [], []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.on('request', lambda req: uploads.append(req.url) if req.method not in ['GET', 'HEAD'] else None)
    page.add_init_script('Object.defineProperty(document,"modelContext",{value:{registerTool(t){window.asterTestTool=t;}}});')
    page.goto('http://127.0.0.1:8765', wait_until='domcontentloaded')
    page.screenshot(path=str(OUT / 'initial.png'), full_page=True)
    page.locator('#files').set_input_files([str(CORPUS), str(TOKENS)])
    page.wait_for_function('document.querySelector("#notice").textContent.includes("2ファイル")')
    assert page.locator('#before').inner_text().strip()
    assert page.locator('#after').inner_text().strip()
    page.locator('#domain-filter').select_option('python')
    page.locator('#split-filter').select_option('train')
    assert page.locator('#sample-select option').count() == 2
    assert '```' in page.locator('#after').inner_text()
    page.locator('#hypothesis').fill('説明とコードのまとまりを確認する。')
    page.locator('#save-note').click()
    assert '保存しました' in page.locator('#note-state').inner_text()
    page.screenshot(path=str(OUT / 'corpus.png'), full_page=True)
    page.locator('[data-tab=tokens]').click()
    model = load_tokenizer(MODEL)
    for text in ['猫は、窓のそばにいる。', 'def add(a, b):\n    return a + b\n', '😀e\u0301\n全角　空白', '']:
        page.locator('#input-text').fill(text)
        actual = page.evaluate('state.ids')
        assert actual == model.encode(text), (text, actual)
        assert '一致' in page.locator('#roundtrip').inner_text()
    page.locator('#input-text').fill('文章を、少しずつ理解する。')
    page.locator('#merge-step').fill('0')
    page.locator('#merge-step').dispatch_event('input')
    assert page.evaluate('state.ids') == list('文章を、少しずつ理解する。'.encode())
    page.locator('#step-all').click()
    page.locator('.token-chip').first.click()
    assert 'ID' in page.locator('#token-detail').inner_text()
    page.screenshot(path=str(OUT / 'tokens.png'), full_page=True)
    # Same app action via a test registry. Native WebMCP is not available here.
    assert page.evaluate('window.asterTestTool.name') == 'aster_inspect_text'
    result = page.evaluate('window.asterTestTool.execute({text:"Aster"})')
    assert result['tokens'] == len(model.encode('Aster'))
    rejected = page.evaluate('()=>{try{window.asterTestTool.execute({text:42});return false;}catch{return true;}}')
    assert rejected
    assert page.locator('#input-text').input_value() == 'Aster'
    page.set_viewport_size({'width':390,'height':844})
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1')
    page.screenshot(path=str(OUT / 'mobile.png'), full_page=True)
    assert not errors, errors
    assert not uploads, uploads
    browser.close()
print(json.dumps({'browser_errors':errors,'uploads':uploads,'python_js_bpe_match':True,'mobile_overflow':False,
                  'webmcp':'test registry validated; native browser support unavailable'}))
