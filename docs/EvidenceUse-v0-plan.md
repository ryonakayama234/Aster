# 次の能力案: 外部情報を根拠に選ぶ

2026-10-14 JST。追跡: [READ #115](https://github.com/ryonakayama234/Aster/issues/115)。
**設計案 / 未実装 / 未凍結 / NOT_RUN**。現在の実験Gate [LANG #113](https://github.com/ryonakayama234/Aster/issues/113) の結果を踏まえて次の仕様を固める。

## なぜこの方向か

Asterの目的は外部の計算・記録・検索・実行を使い、返った情報で処理を進めること。
添付の計画書にあるTokenizer -> TinyLM -> Agent Loop -> Trace/Evaluator -> Memory/Searchの発展を、実装済みのAgent KernelとLANG診断へ結びつける。

偏りを「何の偏りか」で分ける。

| 不足・偏り | この設計が補える可能性 | 残る問題 |
| --- | --- | --- |
| 学習時に知らない固有名・数値・事実 | 実行時の外部記録から参照・コピー | 読み違い、根拠の取り違え |
| 大量の事実を重みに保存できない | 外部storeに保持して必要分だけ渡す | 短いcontextに選んで渡す必要 |
| 関係・指示・構文の学習が不足 | 小さなstructured課題で不足を測定できる | 参照機構だけでは読解を獲得しない |
| 外部資料自体が偏っている | 出典group別評価で失敗範囲を見せる | 資料の偏りは残る |

初期backboneが十分賢いという仮定は置かない。一方、意味を扱う最低限の学習まで不要になるとは主張しない。
初版の固定形式は境界を明確にするための足場であり、自由な日本語理解の完成形ではない。
型や候補器に正解を埋め込み、モデルの理解と呼ぶことを避ける。

## 現在の根拠と判断

- [LANG v0.3結果](LangContextUse-v03-results.md): 独立dev文書1件、追加学習0、TinyLMのbigramに対する差は0.0125869255 nats/token。相対perplexity低下1.250804%は正答率改善ではない。学習前後ともprefix変更へ6/6で反応。元文脈が役立つかは未判定。
- [LANG v0.4設計](LangContextBenefit-v04-plan.md): 保存済み6箇所の符号付き利益から始め、約8未使用prose文書/少なくとも4出典groupを採用する案。量は実施可能性の目安であり、検出力保証ではない。
- [OBS v0c結果コメント](https://github.com/ryonakayama234/Aster/issues/106#issuecomment-6123065099): 計算の後に早期stopした。成功calculator出力とput候補は存在していた。行動失敗であり、容量・Corpus・観測無視のいずれが原因かは未確定。
- [OBS v0d #108](https://github.com/ryonakayama234/Aster/issues/108): この失敗stepのoffline感度診断案。新しいREADの成功や一般化とは別。

Corpusの偏りや容量の小ささは競合仮説であって、失敗原因と確定していない。

## 作業順・達成目標

| 段階 | 作業 | 達成したと判定するもの |
| --- | --- | --- |
| 直近 #113-A | 保存済みprivate reportの同一性確認と全6箇所の利益/害/同等を集計 | 反応の向きが判明; 新しいforwardなし |
| #113-B | 未使用proseの出典・重複・派生確認、manifestと評価規則の凍結 | 独立groupで再現/混在/未観測を記録できる設計 |
| READ設計 | 要求・観測・全候補・正解sidecar・serializer・モデル識別を固定 | コード化可能な仕様; 完成モデルを前提にしない |
| READ G0/G1 | reference/negative control、候補完全性、UTF-8/長さ、ラベル非漏洩 | 測定器の正しさのみ |
| READ G2 | 1段階選択の小さなCPU Trial、全pair保存 | 観測に従った正しい変更か、失敗機構を特定 |

#113は能力PASSの前提試験ではなく、現在地を絞る診断。陽性を待って成功するまで反復するGateにしない。
混在/未観測でも結果を保存し、その失敗範囲を次のREAD設計へ反映する。新たな学習は別の予算/仕様で決める。
本セッションでは私的reportを取得していないので#113-Aは未実施。ユーザーのWSLだけにあるファイルの値を推測しない。

## READ v0で問う一つの能力

同じ要求 `wanted_key=answer` を与える。

| world | 全観測record | 正しい根拠と値 |
| --- | --- | --- |
| A | answer=42 / other=17 | answer側のspan -> 42 |
| B | answer=57 / other=17 | answer側のspan -> 57 |

H1: 根拠の関係を使い両worldに正答する。H0: 固定位置/既知値/formatに依存する、または選べない。
この例はproject-owned structured fixtureであり、ユーザーの日本語Corpusへの合成追加ではない。

初版候補器は全record/value spanを並べる。要求keyと一致するrecordだけに事前に絞らない。
モデルは参照候補を選択する。コピー器は選択spanのbytesを取り出すだけ。正解判定は別verifierが行う。
ルールでkey一致を選ぶreferenceは測定器の対照であり、Asterの学習成果ではない。

```mermaid
flowchart TD
  W["合成world / 実観測"] --> A["全record・value span"]
  Q["要求key"] --> M["モデルの候補採点"]
  A --> M
  M --> C["選択spanのコピー"]
  W --> V["独立verifier"]
  C --> V
  V --> R["選択記録 + Trial結果"]
  A --> R
  M --> R
```

LANG checkpointとDecisionModelは異なるartifact。READのどちらを使用するか/新headを設けるかは設計Gateで選ぶ。
既存Decision入力はAction評価用であり、READへそのまま接続済みとは扱わない。現在のserializer/candidate builderを暗黙変更しない。

## データ契約案

| オブジェクト | 最小の内容 | モデル可視性 |
| --- | --- | --- |
| World | task_id, source_group, split, paired_world_id, observation records | 要求/実観測のみ; group/splitは採点管理側 |
| Candidate | candidate_id, record_ref, UTF-8 byte span, value, token count | 全候補を同じ規則で提示 |
| Selection | all scores, selected_ref, model/tokenizer/serializer IDs | モデル出力; 成功ラベルは含めない |
| Verifier sidecar | worldの正解record/valueと根拠関係 | モデルから不可視 |
| Run | protocol/manifest/source SHA, artifact lineage, resource use, verdict | 実験記録側 |

record_refは出所、spanは元bytesのどこからコピーしたかを示す。値と根拠が同時に合っていることを評価する。
UTF-8境界と実Tokenizer長を測る。context overflowは推論前にBLOCKEDとし、正解に合わせて切り詰めない。

## 測り方・失敗から次の問いへ

Primary候補は「両worldで根拠と値が正しいpair数 / 適格pair数」。答えが変わっただけでは成功にならない。
同一要求・候補数・順序を固定し、各worldの実観測から候補を作り直す。値だけを変えるpairでもtoken長不一致があれば報告し、意味だけの効果と扱わない。

補助対照: 妨害値変更、record順変更、未使用key/value系列。名前やseed違いを独立task familyに水増ししない。
同一generator/templateからの派生は同一splitに置く。上限6程度のdevelopment fixture groupで始める案だが、family定義/件数/予算/判定線は測定前に固定する。
初版は一意の根拠がある課題に限定する。根拠欠落/競合への棄権は後の別能力Gate。

| 観測される失敗 | 次に検証する一つの候補 |
| --- | --- |
| 順番を替えると崩れる | 候補位置shortcut |
| 未使用値で崩れる | 値の暗記か、コピー/Tokenizer境界の問題 |
| 参照選択が一貫してできない | 小さな関係学習でscratchと事前学習初期化をmatched条件で比較する設計 |
| 正しく選べるがAgentで誤用する | 選択済み値をtool引数へ結びつける別Gate |

scratch比較は将来の初期化介入案で、今の固定モデル評価と同時に学習を行うものではない。
単純なルールで解ける課題なら、実用のルールと学習研究を区別し、ニューラルモデルの優位を捏造しない。

## 継続記録の契約

各機能PRに、システム図、データ契約、仮説/比較/達成目標、実装と実測の状態、結果と次の一問を同時に置く。
旧Run/Artifactは不変、新しいcycleをappend。仕様変更後は旧claimを継続しない。
案の状態も記録する。

```json
{
  "cycle_id": "read-evidence-selection-design-v0",
  "status": "proposal_not_run",
  "experiment_mode": "trial",
  "evidence_scope": "design",
  "protocol_frozen": false,
  "measurement_run_id": null,
  "source_git_sha": null,
  "parent_artifact_id": null,
  "result": null
}
```

これは未実行のテンプレート。将来のRunではsource/artifact/manifestを実物で埋め、nullを推測で補わない。
CIのコード確認とprivate WSL能力計測を別に記録する。公開Gitはコード・合成fixture・安全な集計だけ。

担当: エージェント側で仕様・実装・Git/Issue/PR/CI・記録更新を進める。ユーザー側には必要時にproseの選定とWSL private実行を一度に一つ依頼する。
現在はprose新素材の必要性を #113 の設計に沿って整理し、私的モデルへのアクセスを装わない。

機械可読な設計時点の実験台帳は [READ proposal metadata](experiments/read-evidence-selection-v0.json) に置く。`proposal_not_run` は能力計測結果ではない。

## 文献と転用範囲

- [Khandelwal et al., ACL2018](https://aclanthology.org/P18-1027/): LSTM文脈の並べ替え/置換/削除による分析。#113の手法上の参考。
- [Merity et al., Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843): 語彙からの生成と直近文脈からのコピーを組み合わせる。参照/コピーの先行例; このREAD案と同一モデルではない。
- [Lewis et al., RAG](https://arxiv.org/abs/2005.11401): 事前学習readerとretriever、外部Wikipedia indexを使う。外部知識分離の先行例であり、未成熟TinyLMへ検索を付ければ読めるという証拠ではない。
- [Eldan & Li, TinyStories](https://arxiv.org/abs/2305.07759): 限定された英語素材で小さな/単純なモデルの能力を示す。学習課題を限定する参考で、日本語Asterの17.4万parameterモデルへの保証ではない。

Wolframは既存NLL差と指数換算を検算した。計算確認とprivateモデル再測定は別。

## 2026-10-14追補: 偏ったCorpusと弱い初期表現を前提にする

今回の設計判断は、広い日本語Corpusにすべての事実・数値・手続きを覚え込ませる要求を減らし、
まず「要求keyと観測recordの関係を使う」という小さな学習対象へ分けること。
外部記録が補うのは主に事実の保存容量であり、参照関係の学習不足まで自動的に補わない。

| 次の候補 | 初期の能力への依存 | 今の判断 |
| --- | --- | --- |
| 全体のCorpus再配合・追加事前学習 | データ選定、容量、学習量の影響が重なる | LANG診断後の別の一因子研究 |
| 検索を付けた自由回答 | 検索、読解、生成を同時に必要とする | READが使えるかを先に調べる |
| 根拠候補の選択とコピー | 小さな関係選択を必要とし、値の自由生成を要求しない | 次の機能候補として推奨 |
| 長い計画・自己修正・自動再学習 | 観測読解と複数段階の安定した判断が必要 | READ成功後も別Gateとして検証 |

### 「初期構造」と「初期重み」を分ける

入力・全候補・独立verifierという構造は設計者が与える。
要求keyに一致する候補を事前選択する構造にはしない。
この足場は小さな反証可能な課題を作るためのもので、構造なしに知能が成立すると主張しない。

初期重みの言語能力は未確定。固定モデルで選べなければ、
次の別Trialで同じ小型構造を使うscratch初期化と既存LANG初期化の関係学習を比較する設計を作る。
その場合は同じTokenizer、head構成、serializer、教師例、更新予算、評価pairを使い、
初期化以外を揃える。既存DecisionModelとLANGの異なる構造をそのまま比較して初期化効果とは呼ばない。
step数の一致だけで計算量一致とはせず、token提示数・CPU時間・RSSを記録する。
この追補は新head、重み移植、新学習を開始するものではない。

scratchでも未使用値・位置変更へ対応すれば、この限定課題では強い言語事前学習への依存が小さいという観測になる。
両条件が失敗した場合は、入力がcontextに収まっているか、教師関係・学習損失・候補提示が妥当かを先に監査する。
すぐにモデル容量不足と断定しない。scratch成功も一般日本語読解の獲得とは扱わない。

### 小課題で抑える偏りと、残る偏り

project-owned structured generatorでは、要求key、正解候補の位置、値系列を交差させ、
「answerなら常に先頭」「42なら常に正解」というshortcutを抑える案とする。
教師ラベルは独立sidecarから取得し、候補器・serializerへ混ぜない。
同じworldの値違い・順番違いは同一groupへ置く。
未使用値や未使用keyの評価は、同一形式内の汎化probeとして記録する。
未使用generator/templateを評価する場合はtemplate groupを分離し、同一template内probeと区別する。
これらの合成fixtureは日本語prose/dialogue Corpusへの追加ではない。

| probe | 観測例 | 期待すること |
| --- | --- | --- |
| 根拠値変更pair | answer=42をanswer=57へ変更 | 正しいspanと値が両worldで一致 |
| 妨害値変更 | other=17をother=83へ変更 | answerの選択を保つ |
| record順交換 | otherを先頭へ移す | 同じanswer recordを選ぶ |
| 未使用値系列 | 学習にない値を観測へ入れる | 値暗記に頼らず正しい参照を選ぶ |

固定形式は関係学習を測りやすくする一方、形式の偏りは残る。
他の文型・長い文章・根拠欠落・複数根拠・Agent完遂への拡張は、それぞれ別能力として扱う。
外部資料そのものの偏りを、この機構で解消したとは主張しない。

### 達成目標と記録の入口

主な能力判断は両worldで根拠と値が正しいpair率。
referenceと固定位置対照も同じpairで評価し、対照との差、全失敗、条件ごとの件数を併記する。
合格線と必要pair数は結果を見る前に定める。現時点で数値合格線は未凍結。
失敗なら候補欠落・context overflow・参照誤り・コピー誤りを分け、
次の変更を一つ選んで停止する。良い例だけを追加して同じTrialを継続しない。

検索しやすい正本は #113（現在の文脈診断）、#115（READ能力の追跡）、この文書（設計・データ・結果状態）、
[architecture.md](architecture.md)（全体図）。重複Issueを作らない。
実装時には、ここで固定したデータ契約に対応するschema/manifestと、
source SHA・artifact ID・比較条件・結果・限界・次の一問を持つreportを同じPRからリンクする。
本追補の実験記録は `status=proposal_not_run`、`measurement_run_id=null`、`result=null`。

## 後続ロードマップ: Evidence LoopはREADの後で組み立てる（2026-10-15追記）

READ v0は「既に短い観測が与えられたときに、要求された根拠を選べるか」だけを測る。
本プロジェクトの最終目標はより広い **Evidence Loop**（必要な根拠を取得・検証・再利用する）だが、
その完成をREAD v0の実装・成功として記録しない。Corpusの偏り・小型化への対策は、
「外部の事実を持ち込むこと」と「観測を読んで使い分けること」の両方が必要。

```mermaid
flowchart TD
  U["タスク・要求"] --> P["不足情報の検出（将来）"]
  P --> R["外部記録の検索（将来）"]
  R --> O["短く、出典付きのObservation"]
  O --> S["READ v0: 全候補から根拠を選ぶ（次の能力Gate）"]
  S --> C["選択した値・source spanを保持"]
  C --> V["独立Verifier（v0は実験の外側）"]
  V --> J["Trace / Runへ記録"]
  J -. "別のGateで検証後" .-> M["必要に応じた再検索・外部Memory（将来）"]
  M -. "再観測" .-> O
```

追加設計は段階的に行う。まず **Phase R: READ**（モデルによる根拠選択）を独立評価する。
次に **Phase S: Search**（必要な短い証拠を探す。検索不能も結果として返す）、
**Phase V: Verify**（証拠と答えの整合、古さ、矛盾を分けて検査）、
**Phase M: Memory**（正本/source revisionと有効期間を伴う再利用）、
最後に **Phase A: Agent**（未知状態での再探索・行動変更）と進む。
各Phaseは別のIssue/Trial Gateを要し、現行の #113 / #115 を上書きしない。

### 将来の統制比較（現在は設計メモ、NOT_RUN）

| 条件 | 与えるもの | 切り分ける原因 |
| --- | --- | --- |
| A: 閉じたモデル | 要求のみ | パラメータ内知識だけで足りるか |
| B: 全文渡し | 同じ外部記録の全内容 | 過長contextで失敗するか |
| C: Oracle抜粋 | 正しい根拠だけを外部から選んで渡す | 読解・回答側の上限（Aster検索能力と数えない） |
| D: 自動検索 | 同じ記録集から検索し、独立READで選ぶ | 検索失敗と根拠選択失敗を分離 |
| E: D + 検証・再試行 | 明示budget内で再取得できる | 追加試行の利益・副作用・コスト |

A–Eを同じTask / document-family単位、同じ目標・出典・評価器で比較する。
検索対象に答えが無いケース、誤った資料、相反する資料、未知の値、順序変更を含め、
正答率だけでなく誤根拠率・根拠欠落時の誤断定率・検索回数・token長・時間を残す。
信頼できる外部情報が存在しなければ回復は保証できない。初期の固定形式・oracleは
測定の道具であって、構造化されていない会話能力の証明ではない。

### 外部研究との位置関係（Aster実測ではない）

- [ReAct (ICLR 2023)](https://arxiv.org/abs/2210.03629): 推論・外部行動・観測を組み合わせる先行研究。Asterの現行ModelPolicyへそのまま移植済みではない。
- [Self-RAG (ICLR 2024)](https://arxiv.org/abs/2310.11511): 必要なときに検索し、自ら証拠を批評する学習済みLMの設計。現在のTinyLMに同機能があるとは仮定しない。
- [Corrective RAG (2024)](https://arxiv.org/abs/2401.15884): 検索結果の品質が低い場合の補正を扱う。検索を付けるだけでは不十分という参考。
- [Lost in the Middle (TACL 2024)](https://aclanthology.org/2024.tacl-1.9/): 長い文脈を渡すだけでは証拠が利用されない場合がある。全文投入との比較を置く理由。

**次の一問は変えない**：まず #113 の保存済み文脈診断を確認し、READ v0が
「要求と観測の関係」から根拠を選べるかを独立に測る。検索・Memory・自動学習はまだ作らない。
