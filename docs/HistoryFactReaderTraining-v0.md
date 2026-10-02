# 履歴reader：固定課題の学習・実行診断 v0

2026-10-02 JST。PR #42 head 225454935761b6e3e01043be5a93ba56e0f95a9cを基準。

## 考え方・仮説・反証条件
実観測事実→行動種類が成立し、文字列→事実の課題生成も成立した。
今問うのは、同じ実履歴からC/M/Fを学習して読み、固定分類器へ渡せるか。
仮説は固定小型readerが学習例を安定fitし、task単位のdevでも状態対照と実行を保持すること。
train未fitなら汎化だけの問題と呼ばない。fit後dev対照/完了が崩れれば未知taskの読解を再検討する。
事前学習効果は今回問わない。random initializationからの課題用学習のみ。

## 固定条件
PR #42のtrain80/dev40、suite e6f3c7f7fe3f9703b5f46591471e3e45142e304451c74bce375cd84da282fbe1。
入力schema/全イベント/分割は変更しない。旧/予約test生成・点検・採点0。
UTF-8 byte0..255+BOS256+EOS257、vocab258、context2048、TinyLM width16/heads2/layers1。
EOSの表現からLinear16→9、C/M/F各3クラス（false/true/unknown）。LM headは凍結して未使用。
特殊tokenと位置も含めrandom initialization。Byte列単例、paddingなし、全文因果attention。
AdamW lr0.001、default betas/eps/weight_decay、gradient clip1.0。
50epoch×80単例=4,000更新/seed、seed42/43/44。独立shuffle generatorをseedで固定。
最終checkpointだけをdev評価。毎epoch末train全80をeval/no_gradで評価。
train fit=最後3epochすべて3事実同時80/80かつ全事実target margin>0。
CPU2threads、各seed学習+reload+評価900秒deadline。時間超過・失敗も保存し延長しない。
開始前にtrainのみ5updateの速度確認。結果でLR/epochを選ばず、速度が上限に収まらなければ本測定前にprotocol改版。

## 固定した外部行動分類器
PR #41正式Run212b32fd029640ce8efc89930b5c4403/seed42の保存checkpointを再利用。
model digest de12ec2dedfac982d3085ccb24540b26fcf20eecb1d0cf06f99c31aad9efea32、
weights file e5ce5af484beb59fc5cbc6c6a640a23b34d0772d6482cc48caf5bf47ca491b87。
全reader seedで同一、更新0。既存feature/binder/goal契約を要求。再学習しない。
候補生成と引数結合は外部実装。readerは算術/引数生成/依頼文のTask化を担当しない。

## unknownとabstain
unknownは「履歴に該当観測がない」という事実ラベルで、不確信確率の閾値ではない。
生成preflightにある6組だけを今回の有効scopeとし、それ以外の予測組はabstain。
合法unknownは既存feature契約どおり0へ写す（例：未計算→calculator、未取得→get）。
C=unknown、C=falseなのにM/F=true等はabstainし、正解事実で穴埋めしない。
確率は未校正。confidenceによる自動採用はしない。

## 対照・測定
- 正解事実→固定分類器→既存候補結合の全120prefix継続。これをoracle上限として一度確認。
- reader予測事実→同分類器→候補結合の全120prefix継続、train80/dev40を分離。
- 初期normal0の8train/4devはprefix測定の部分集合として別表示し件数に重複加算しない。
- train由来の各事実多数クラスの対照（非学習・履歴不使用）。train/dev事実完全一致を保存。
- 3事実同時一致、事実別3×3混同行列/positive precision-recall/unknown件数、candidate結合とAction一致を分離。
- dev状態対照16組の両方事実正答、family別結果、誤停止/誤保存/abstain/binding_failure/invalid、最初の誤読を記録。
- teacher-prefixと自分の行動で訪れた状態を分離。現状態の真値は記録・採点専用で選択へ渡さない。
- 停止時memoryと最後対象put後getを独立に再確認し、Evaluatorのtask_successと照合。
- visited context超過はinvalidとして失敗と分離。両経路の共通評価可能分母も保存。

## 保存・停止・次の分岐
source/Python/protocol/登録suite hash、初期・最終weight、全shuffle、曲線/予測/実trajectoryをRunへ保存。
readerの保存/reload全logits一致、evaluation前後reader/classifier weightとRNG不変を検証。
checkpointはinput/classes/config/goal/classifier契約とfile hashを検査。optimizerなし、resume非対応。
正式3seedを一度だけ測定して終了。debugと速度確認は正式へ加算しない。
保存cached予測とAction/実Tool再生の独立監査を追加し、誤りも隠さずreportsへ残す。
fit未成立→固定データで最適化/表現の診断。fit/dev失敗→一致・対象・順序の読解に分解。
fit/dev/実行成功→別familyを事前登録して構造holdout、次に短い依頼→Taskへ。
random/pretrained比較は素材/Tokenizer/architecture/contextの整合後に別protocolで行う。

devはshared templateのparameter transfer、3seedを独立task数に足さない。
公開Service/API/aster-web/モデルpromotionなし。ユーザー側追加作業不要。

## 正式測定前の配線・資源確認
CPU2threads/torch2.14.1+cpu。5短train更新0.023秒、5最長train更新0.048秒、最大1005token。
全pytest184 passed、Pyright0 errors/0 warnings。速度確認中は検証併走で精密benchではない。
1epoch debug Run7295584c020141ff90064041cffadaacは監査済みで正式3seedへ加算しない。
学習条件は変更せず、正式測定へ進む。更新数はWolframで4,000/seed、12,000/3seed、
train評価12,240行、dev対照16組/seed、自力prefix360episodeと確認した。
