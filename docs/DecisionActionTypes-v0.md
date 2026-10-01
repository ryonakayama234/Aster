# 観測特徴→行動種類の学習診断 v0

2026-10-02 JST。PR #40（40ca7fc、local64b5322とtree一致）を基準。

## 考え方・仮説・達成目標

関係をJSON文字列へ書いても未知状態対照は改善しなかった。次は文字列/具体値/候補引数を
学習器から外し、観測済み事実からAction種類を学習できるかを診断する。
これは表現/モデル/出力分解の複合変更であり、前実験との純粋な因果比較ではない。
仮説: 3つの数値特徴から4種類を選ぶ線形分類器は、既知task保持と数値/キーを変えた
自力完了を両立できる。全3seedのtrain stable fit/既知保持と全評価prefix/初期完了を要求。
失敗なら抽出/教師/結合/学習更新を再点検。成功なら文字列読解を独立した後続へ切り出す。

## 入力・モデル・結合

C=現taskの成功calculator観測がある。M=現在の対象memoryがその観測値と一致する。
F=最後の対象putより後の成功getがあり、その返値が観測値と一致する。
既存observable_relationsからのみ導出。未観測M/Fは0へ写す。Cがない間M/Fを使う能力は測らない。
teacher/Evaluator/独立計算期待解を入力へ渡さず、具体値/キー/候補自体も学習器へ渡さない。
抽出済み観測事実・equality・時系列処理は外部実装の能力として帰属する。
3→4のnn.Linear（weight12+bias4=16 parameter）、ランダム初期化。
出力順calculator/memory.put/memory.get/stop。teacherのAction種類だけを教師にする。
選択種類の既存candidateが一つならそれへ結合。stopはgoal_verifiedがあればそれ、
なければ唯一policy_stopへ結合。引数はcandidate builderが供給する。
種類が候補にない/同種類に複数候補なら結合失敗として記録し、rule fallbackしない。
最後Tool失敗は診断の範囲外として明示拒否（invalid）；一般的失敗復帰は主張しない。

## 固定学習・評価条件

PR #39/40のnormal-plus-recovery-v1 48raw判断/64slotを使用。
seed42/43/44、25epoch、64単例更新/epoch=1,600/seed、AdamW lr0.01/既存default、2threads。
各epoch全48のNLL/accuracy/margin、提示順、初期/最終weightを保存。最終3epoch全正答+margin>0をfitとする。
既存128 dev prefixと32状態対照、各variant初期2episode、model-only最大8stepを測る。
追加devの初期4taskを学習前に固定: add(123,-45)→k_delta、add(-7,0)→aux_total_v1、
subtract(100,123)→diff_new、subtract(-4,-9)→slot_z。teacher rolloutを検証してから使う。
これらも同じtask familyの数値/キー移送probeで独立holdoutではない。旧/予約testは触らない。
特徴空間のtrain重複を全caseで計算。重複なら「未知特徴への汎化」と呼ばず、
具体値/名前を外部処理したequivalence transferとして解釈する。raw入力非重複分母を流用しない。

## 事前登録・検証・停止

48train/128dev/4新taskの実再生、教師coverage、features→typeの衝突0、feature range、
metadata非依存、候補逆順の結合不変をpreflightで検査しdigestを正式runnerに要求。
重みの保存/reload、feature/binder/goal契約mismatch拒否、評価中weights/RNG不変を確認。
保存全score/特徴/選択/実Action/trajectoryを別監査CLIで再計算/Tool再実行し、
最後対象put後getと停止時memoryを独立に検査する。各seed5分deadline、失敗も保存。
3seed固定、一度測定して終了。改善へ条件を後付け変更しない。

達成は再現可能な3seed診断と次の境界の確定。成功しても生履歴読解/会話/コード生成を獲得したとは言わない。
main/Service/API/aster-web/公開モデル/promotionは変更しない。ユーザー側追加作業不要。
