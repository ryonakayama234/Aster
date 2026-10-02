# Asterで一緒に開発するエージェントへ

## 目的

Asterは、自作TinyLMを通してAI・Python・数学・コンピューターサイエンスを学ぶプロジェクト。
ユーザーが仕組みを理解し、自分で変えられることを成果に含める。
将来の中心は外部化知能：計算、記録、検索、実行を外部の道具へ委ね、結果を使って行動できること。

## 協働の進め方

- 日本語で説明する。概念の直感、小さな具体例、実装、観測結果の順に結びつける。
- 数式や専門語は必要なときに使い、記号の意味を説明する。
- 技術的な選択肢は推奨案と理由を示す。ユーザーに未習得のアルゴリズム選択を丸投げしない。
- 変更は理解できる単位に区切る。学習の節目で「今回わかったこと」「次に確かめること」を短く残す。
- 学習目的を理由に、頼まれた作業を止めたり、必須の小テストを課したりしない。
- 事実、設計案、未実装、実験による確認を区別する。テスト成功を知能や研究成功の証拠にしない。

## GitHub・外部計算資源のコスト

- 明示的な依頼なしに、従量課金が発生し得るGitHub機能や外部計算資源を有効化しない。
- `.github/workflows/**` を変更する場合は、原則としてGitHub標準runnerを使う。
- larger runner・GPU runner・有料runnerを、明示的な確認なしに使用しない。
- Codespaces、GitHub Models、cloud agentなどの有料・従量課金機能を、明示的な確認なしに新規利用しない。
- 不要なmatrix build、scheduled workflow、大容量artifact保存、過剰なcacheを追加しない。
- CI変更時は実行頻度、runner、artifact/cache、外部service利用を確認し、コスト増加の可能性があれば事前に明記する。
- 無料枠内であっても、同じ検証を重複実行しないようworkflowのtriggerとconcurrencyを設計する。

## コーパスの所有と由来

- 日本語prose・dialogueは、ユーザーの自作文と、ユーザーが選んだ外部文章・AI対話を候補に含める（2026-09-20更新）。明示的な依頼なしに代筆・外部収集・合成して埋めない。
- ユーザーが選んだことと、ユーザーが書いたことを区別する。出典・作者・AI生成元・確認状態を記録し、読む素材と応答のお手本を分ける。
- Math・Pythonは出典と利用条件を確認した外部資料から始める。
- structured・JSON・toolはAsterプロジェクト内で作る。外部tool会話コーパスで置き換えない。
- 初期のプログラム生成と、学習後のAsterモデルによる生成を区別して記録する。
- モデルが書いたtool結果を実行済みの観測として扱わない。正解は検証器・実行器で確認する。
- 原本、出典、ライセンス、revision、ハッシュ、変換、分割、除外理由を追跡する。
- 評価用データをBPE学習にも混ぜない。元問題・章・生成テンプレートの派生は同じ分割へ置く。

## 実装と実験

- 方針案は `docs/corpus/AsterCorpus-v0.md`。案の段階を勝手に凍結済みと扱わない。
- v0を固定してTokenizerとTinyLMで評価した後、v0.1へ変更する。改善案をv0へ静かに混ぜない。
- Corpusの版とTokenizerの版は別。既存AsterTokenizer-v0.1がCorpus-v0を使っても矛盾しない。
- まず既存の単純なbyte-level BPEを使う。速度や品質に問題が出たら計測して改善する。
- Decision課題のTokenizer/入力表現比較は、一般Corpus用AsterTokenizerの置換と混同しない。同じdecision内容・candidate・教師・更新順を固定し、保存済みTokenizer artifactのdigest、系列長、padding込みtoken量、parameter数、CPU費用を記録する。
- Decisionのsealed test / reserved testは、比較条件を決めるためのTokenizer学習・長さ確認・scoreへ流さない。debug runとpreregistered measurement、ChatGPT/GitHub runner測定とユーザーWSL測定を区別する。
- 語彙数が異なるモデルを比較するとき「同じseedだから同じ初期weight」とは扱わない。共有したtensor/semantic rowと共有できない語彙依存weightを分け、hashで残す。
- このWSLチェックアウトではLinux側のPython・Gitを使う。ユーザーの未コミット作業を保護する。
- 既存の読み込み対象拡張子・特殊トークン・文書境界を確認してから接続する。
- 外部コードを収集時に実行しない。将来の実行器はネットワークなし・隔離・時間/資源制限を設ける。
- 変更に見合う確認を行い、未検証部分を明記する。大きな依存や抽象化を先回りして入れない。

## Sites変更時の継続契約（2026-09-27合意）

- Asterを操作・観測・育成するSitesに関係する変更では、最初に `docs/ServiceContract-v0.md`（存在するブランチ/PR）、`docs/UI/SitesWorkbenchContract-v0.md`、`docs/UI/SitesWorkbench-changelog.md` を読む。
- 同じ作業内で設計契約を現状へ更新し、changelogへ変更理由・ソースcommit/PR・検証結果・未検証・公開状態・次の課題を追記してGitへ保存する。
- API契約とUI責務を区別し、Job/Run/Artifactの識別、出典・採用理由、実測と保存再生の区別を保つ。
- 将来案は未実装と明示する。前提PRが未マージなら導入済みと扱わない。設計書を完成記録だけ先に更新しない。

## 反証可能な研究の継続契約（2026-09-30）

- 各実験の実装・測定前に「考えたこと/観測根拠」「仮説と反証になる結果」「固定条件/変更点/対照」「達成目標/停止条件/結果別の次の分岐」をdocsへ記録する。
- 配線成功、train fit、dev対応、自力episode完了、独立holdoutを分ける。仕組みの追加を研究成果の代用にしない。
- 条件変更は測定前に改版する。改善なしや悪化も証拠と判断を残して終了できる。
- こちらで可能な実装・検証・記録は進め、ユーザーPC/私的素材/本人の選定が必要な役割だけ、目的・具体手順・返してほしい成果物を提示する。
- 実測後は同じ作業でreportsへ全条件・未取得・限界・次の分岐を残す。言語能力と行動能力の成果を混同しない。

2026-10-01補足: 各段階の着手前に考え方・作業方法・達成目標をユーザーへ短く共有する。入力表現比較では同じseed内の全初期重みと提示順を固定し、serializer IDをcheckpointの検証対象にする。新しいdev対照組を、同じ実験のtrainへ混ぜない。独立holdoutでない結果は診断として記録する。

2026-10-01状態train補足: データ変更は実履歴だけでなくmodel-visible入力/候補順非依存署名で検査する。compactで消えるdelayを新学習内容と扱わない。冗長計算prefixの予約test生成familyをtrain/devへ移さない。評価入力は比較全armのtrain入力unionと照合し、同一入力は保持確認として別集計する。正常例の置換と追加保持を区別し、状態判断・自力完了・元課題保持・数値変更を別々に報告する。

2026-10-01完了契約合意: Asterの復帰課題は「停止時点でも正しく保存され、最後の対象キー変更後に取得確認できている」を目指す。正本はdocs/CurrentStateGoal-v1.md。新契約の教師・候補・評価器を同じIDで選択し、旧v0の実測をv1として扱わない。仕様テストの成功と学習済みモデルの復帰能力は別に報告する。

2026-10-01固定v1診断補足: 旧v0モデルをv1へ診断する際は仕様変更による分布差を明記し、教師/候補/評価/再生をv1で揃える。候補集合が供給する情報を監査し、同候補・異targetの対照を報告する。候補1個のmarginは未定義とし、teacher Action一致と停止時のtask_successを分ける。

2026-10-01混合計画補足: docs/DecisionMixedTraining-v0.mdを実装する際は、v1教師/候補/評価を揃え、正常例を全て含むことと個別exposure一致を区別する。新train unionとdev入力を再照合し、旧非重複件数を流用しない。計画保存・生成preflight・能力測定を別の達成として報告する。

2026-10-01混合preflight補足: 生成正本はreports/decision-mixed-v1-preflight-v0.json、source/slot/suite/input/Tokenizer digestを次runner開始前に一致検査する。初期空履歴は既存Trajectory互換のv0識別になるため、v1 Session/Run契約を明示し非空履歴をv1へ揃える。preflight成功と学習能力を区別する。

2026-10-01混合実測補足: reports/decision-mixed-v1-study-v0.jsonを全6条件の正本とする。train fit/既知保持/非重複状態対照/途中prefix/初期完了/invalidを区別する。主対照は改善せず、prefixのseed依存改善と混合seed44の数値1/2を省略しない。次の関係明示入力は未実装。観測Tool結果からのみ導出し、教師Action・Evaluator goal・正解値・task段階ラベルを入力へ漏らさない。追加測定前に別protocolを登録する。

2026-10-01関係入力比較補足: compact/relationsの比較はdocs/DecisionRelationsComparison-v0.mdと登録preflight digestへ従う。関係抽出・履歴正規化・系列長の複合介入を純粋な特徴効果と呼ばない。各表現のtrain重複を再計算し、共通非重複の主対照を使う。抽出はraw観測から独立scanし、外部化した関係判断をモデル能力に算入しない。新serializerのcheckpointをcompact既定で読み込ませない。

2026-10-02行動種類診断補足: docs/DecisionActionTypes-v0.mdに従い、特徴train重複を必ず記録する。C/M/F直接入力と既存candidate結合を使う成功を、生履歴読解/算術/引数生成/未知特徴への汎化へ帰属しない。種別候補欠落は結合失敗として記録しrule fallbackしない。言語後続はdocs/LanguageToActionRoadmap-v0.mdで次token事前学習と事実抽出課題学習を区別する。

2026-10-02履歴reader補足: docs/HistoryFactReader-v0.mdに従い、入力へcandidate/teacher/evaluation/stepを混入させない。unknownとfalseを区別し、固定行動分類器への写像と事実読解を別測定にする。task単位分割と共有templateを併記し、実Tool生成/独立oracle検証をモデル能力と呼ばない。

2026-10-02reader学習補足: docs/HistoryFactReaderTraining-v0.mdを正式測定前に固定。未知観測unknownと予測矛盾abstainを区別し、保存済みseed42分類器を全reader seedで固定する。予測事実を真値へ差し替えず、誤停止/abstain/invalidを保存し、train fitとdev読解と自力完了を別報告。
