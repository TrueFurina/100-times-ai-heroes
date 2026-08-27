# Issue #10: ローカル画像モデルのプロファイル化と比較ベンチマーク

## 背景

このプロジェクトの画像要件は、単なる高速プレビューではなく、JRPG・アニメ系の全身キャラクター作画である。Animagine XL 4.0 Optは有力な初期候補だが、Illustrious XL v2.0、Pony Diffusion V6 XL、NoobAI XL 1.1にも用途上の優位性があり、単一モデルを事前に決めてはならない。

## 目的

候補画像モデルを同じ固定ケース・固定seedで比較し、品質、生成時間、メモリ使用量を根拠に採用モデルを決められるようにする。

## 対象モデル

| profile ID | checkpoint | 用途 |
|---|---|---|
| `animagine-xl-4.0-opt` | `animagine-xl-4.0-opt.safetensors` | アニメ・JRPG系の第一候補 |
| `illustrious-xl-v2` | `Illustrious-XL-v2.0.safetensors` | 線・色を重視するイラスト |
| `pony-v6-xl` | `ponyDiffusionV6XL_v6StartWithThisOne.safetensors` | 獣人・異種族 |
| `noobai-xl-1.1` | `NoobAI-XL-v1.1.safetensors` | 実験的な品質候補 |

モデルは自動ダウンロードしない。利用者がComfyUIの`models/checkpoints`へ配置する。各モデルの利用条件はモデルカードを確認する。

## 実装仕様

### 1. モデルprofile

`config/comfyui/model_profiles.json`を追加し、各profileに次を定義する。

- checkpoint filename
- prompt style
- width / height
- steps
- CFG
- sampler / scheduler
- negative prompt
- clip skip（必要なモデルのみ）
- source URL、license、experimentalフラグ

`COMFYUI_MODEL_PROFILE`でprofileを選択する。`COMFYUI_CHECKPOINT_NAME`、解像度、steps、CFGなどの個別環境変数は設定した場合だけprofileを上書きする。

### 2. モデル固有prompt

LLMが生成したconceptをそのまま長文promptとして渡さず、profileごとのprompt builderを通す。

- `gender`から`1girl`、`1boy`、`1other`を設定する
- `age`、`species`、`role`、`ability`を視覚属性として含める
- `solo`、`full body`、`standing`、`feet visible`、`white background`を共通指定する
- Animagine/Illustrious/NoobAIはモデルの推奨品質タグを使う
- Ponyは`source_anime`とPony固有のscoreタグを使う
- Ponyの`clip skip 2`をworkflowへ適用する

### 3. ベンチマーク

`tools/benchmark_image_models.py`を追加する。

既定の固定ケースは次の3件とする。

1. 人間の若い男性剣士
2. 半人半水生の女性デザイナー
3. 中年男性ライカンスロープのタトゥーアーティスト

既定のseedは`101,202`とする。`--profiles`、`--cases`、`--seeds`で変更できる。

ベンチマークは次を満たす。

- 同じcaseとseedを全profileへ渡す
- batch sizeは常に1
- 各画像のprompt、profile、seed、経過秒数を記録する
- macOSでは`memory_pressure -Q`を生成中にサンプリングし、最低空きメモリ率を記録する
- `--dry-run`ではComfyUIへ接続せず、profileとpromptだけ検証する
- profile切り替え時は既定でComfyUI `/free`を呼ぶ
- `data/model_benchmark_<timestamp>/report.json`へ結果を保存する

### 4. メモリ安全性

- モデルを並列ロードしない
- 画像生成は1枚ずつ行う
- Hires.fix、アップスケール、LoRA、ControlNetを比較workflowへ含めない
- 生成中の空きメモリ率が設定閾値を下回った場合は新規生成を停止する
- 832x1216で失敗した場合に限り、768x1152へ1回だけフォールバックできる
- フォールバックや停止理由をreportへ記録する

## 受け入れ条件

- `python tools/benchmark_image_models.py --dry-run`が外部通信なしで完了する
- 4つのprofileがJSONから読み込める
- profileごとにcheckpoint、prompt形式、推奨パラメータが異なる
- Pony profileではclip skipがworkflow node 11へ反映される
- 同じseedが全profileで使われる
- 実ComfyUI環境で、導入済みprofileについて画像と`report.json`が作成される
- reportに成功・失敗、経過秒数、メモリ計測値が残る
- ComfyUI停止時でもベンチマークがクラッシュせず、接続エラーをreportへ記録する
- 既存の`pytest -q`が成功する

## 比較時の採点基準

各画像を0〜2点で採点し、合計点だけでなく致命的な失敗数も記録する。

- 全身と足元が収まる
- キャラクターが1人だけ
- 顔、手、四肢が破綻していない
- species、role、abilityが衣装・小道具に反映される
- 白背景と画風が維持される
- 文字、署名、透かし、重複人物がない

## 対象外

- モデルファイルの自動ダウンロード
- クラウド画像生成API
- 自動で最終モデルを決定する画像評価AI
- Hires.fix、アップスケール、ControlNet、LoRAの品質比較
