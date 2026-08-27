# 100 Times AI Heroes ローカル版設計仕様

## 1. 目的と実行方針

この版は、キャラクター設定、テキスト生成、データ保存、任意の画像生成をローカルで完結させる。

- データ保存: ローカルCSV
- 既定のテキスト生成: localhost上のOllama
- 任意の画像生成: localhost上のComfyUI
- Google Sheets: 使用しない
- 既定の実行でのクラウドAPI: 使用しない
- 任意のクラウドLLM: `--provider openai` を明示した場合だけ使用する

Ollamaや画像モデルの初回ダウンロードは事前準備として手動で行う。生成プログラムはモデルを自動ダウンロードしない。

## 2. 処理フロー

```text
data/seed_*.csv
       │ ランダムに属性を選択
       ▼
LLM provider（既定: Ollama localhost）
       │ concept / name / profile / catchphrase
       │ new_ability / new_wants / new_role
       ▼
image_promptを作成
       │ --generate-images指定時のみ
       ▼
ComfyUI localhost
       ▼
data/run_日時/
  ├── output.csv
  ├── errors.jsonl（失敗時のみ）
  └── images/（画像生成時のみ）
```

## 3. 設定

`.env.example`を`.env`へコピーして使用する。

### 3.1 必須のローカル設定

| Name | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` または `openai` |
| `OLLAMA_MODEL` | `gpt-oss:20b` | Ollamaで使うモデル |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollamaのlocalhost URL |
| `DATA_DIR` | `./data` | seedと実行結果の保存先 |

### 3.2 ローカル画像生成設定

| Name | Default | Description |
|---|---|---|
| `GENERATE_IMAGES` | `false` | CLI未指定時の画像生成可否 |
| `COMFYUI_URL` | `http://127.0.0.1:8188` | ComfyUIのlocalhost URL |
| `COMFYUI_MODEL_PROFILE` | `animagine-xl-4.0-opt` | 画像モデルprofile ID |
| `COMFYUI_MODEL_PROFILES_PATH` | `./config/comfyui/model_profiles.json` | profile定義JSON |
| `COMFYUI_WORKFLOW_PATH` | `./config/comfyui/text2image_api_workflow.json` | API形式workflow |
| `COMFYUI_CHECKPOINT_NAME` | profile依存 | 設定時のみprofileのcheckpoint名を上書き |
| `COMFYUI_TIMEOUT_SECONDS` | `300` | 1画像の最大待機秒数 |
| `COMFYUI_POLL_INTERVAL_SECONDS` | `1` | history確認間隔 |
| `MEMORY_GUARD_ENABLED` | `true` | 画像生成前のシステムメモリ安全弁 |
| `MINIMUM_AVAILABLE_MEMORY_PERCENT` | `15` | 次の画像生成を開始する最低空きメモリ率 |
| `COMFYUI_WIDTH` | profile依存 | 設定時のみprofileを上書き |
| `COMFYUI_HEIGHT` | profile依存 | 設定時のみprofileを上書き |
| `COMFYUI_STEPS` | profile依存 | 設定時のみprofileを上書き |
| `COMFYUI_CFG` | profile依存 | 設定時のみprofileを上書き |
| `COMFYUI_SAMPLER` | profile依存 | 設定時のみprofileを上書き |
| `COMFYUI_SCHEDULER` | profile依存 | 設定時のみprofileを上書き |
| `COMFYUI_NEGATIVE_PROMPT` | profile依存 | 設定時のみprofileを上書き |

画像モデルprofileは`config/comfyui/model_profiles.json`で管理する。候補は次の4つである。

| Profile | 主な用途 | 備考 |
|---|---|---|
| `animagine-xl-4.0-opt` | アニメ・JRPG系の標準候補 | tag形式、832x1216、28 steps、CFG 5 |
| `illustrious-xl-v2` | イラスト性・線・色 | tag形式、832x1216、30 steps、CFG 7 |
| `pony-v6-xl` | 獣人・異種族 | `clip_skip=-2`、利用条件確認が必要 |
| `noobai-xl-1.1` | 実験的な品質候補 | 実サービス検証後に採用可否を決める |

モデルは単一候補に固定せず、同じ入力とseedで比較して採用モデルを決める。

### 3.3 任意のクラウドLLM設定

| Name | Default | Description |
|---|---|---|
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI用モデル |
| `OPENAI_API_KEY` | empty | `--provider openai`時のみ必要 |

クラウドLLM用の依存関係は`requirements-cloud.txt`に分離する。

## 4. seed CSV仕様

seed CSVはすべてUTF-8、1列で、1行目がヘッダーであること。

| Path | Header |
|---|---|
| `data/seed_age.csv` | `age` |
| `data/seed_gender.csv` | `gender` |
| `data/seed_species.csv` | `species` |
| `data/seed_ability.csv` | `ability` |
| `data/seed_wants.csv` | `wants` |
| `data/seed_role.csv` | `role` |

2行目以降の空行は無視する。有効な値が1件もない場合、生成開始前に`SeedDataError`を送出する。

生成後に、`ability`、`wants`、`role`の各CSVへ対キャラクター用の値を1件ずつ追加する。重複は許容する。

## 5. LLMプロバイダー

生成処理は、次の共通インターフェースだけを利用する。

```python
generate(prompt: str, max_retries: int = 3) -> str
```

### 5.1 Ollama

- 初期化時に`list()`でモデルの存在だけを確認する
- 未導入でも`pull()`は呼ばない
- 未導入時は`ollama pull <model>`を案内して終了する
- 推論は`chat()`、最大出力2048、temperature 0.8
- 推論失敗は最大3回、2のべき乗秒で再試行する

### 5.2 OpenAI（任意）

- `--provider openai`を明示した場合だけ初期化する
- OpenAI Python SDKのResponses APIを使う
- `OPENAI_API_KEY`がない場合は推論開始前に終了する
- APIキーをログ、CSV、errors.jsonlへ書き込まない
- `store=False`を指定する
- 画像生成はOpenAIではなくComfyUIを使う

## 6. ローカル画像生成

ComfyUI API形式のworkflowを`config/comfyui/text2image_api_workflow.json`で管理する。

必須ノードIDと役割は次のとおり。

| ID | class_type | 書き換える値 |
|---|---|---|
| `4` | `CheckpointLoaderSimple` | `ckpt_name` |
| `5` | `KSampler` | `seed`, `steps`, `cfg`, `sampler_name`, `scheduler` |
| `6` | `CLIPTextEncode` | positive prompt |
| `7` | `CLIPTextEncode` | negative prompt |
| `8` | `EmptyLatentImage` | width, height, batch_size=1 |
| `9` | `VAEDecode` | 固定接続 |
| `10` | `SaveImage` | 固定接続 |
| `11` | `CLIPSetLastLayer` | profileが要求する場合のclip skip |

API処理は次の順番で行う。

1. `POST /prompt`でworkflowをキューへ追加する
2. 返された`prompt_id`を取得する
3. `GET /history/{prompt_id}`を1秒間隔で確認する
4. `status.status_str=success`になったら最初の画像情報を取得する
5. `GET /view?filename=...&subfolder=...&type=...`で画像を取得する
6. `data/run_日時/images/001_<safe-name>.png`へ一時ファイル経由で保存する

`--generate-images`未指定時はComfyUIへ接続してはいけない。

画像生成は常にbatch size 1で行う。複数profileの比較時は、profile切り替え前にComfyUIの`POST /free`でモデルを解放する。大量生成では、OllamaとComfyUIを並列実行してはならない。

## 7.1 モデル比較ベンチマーク

`tools/benchmark_image_models.py`を使い、固定ケースと固定seedで候補を比較する。

```bash
# 外部サービスなしでprofileとpromptを確認
python tools/benchmark_image_models.py --dry-run

# ComfyUIで実画像を生成し、report.jsonへ結果を保存
python tools/benchmark_image_models.py --cases 3 --seeds 101,202
```

比較結果では、少なくとも次を人手で採点する。

- 全身と足元が画面内に収まっているか
- キャラクターが1人だけか
- 手、顔、四肢の破綻が少ないか
- species、role、abilityが衣装・小道具へ反映されているか
- 白背景とJRPG系の画風が維持されているか
- 文字、署名、透かし、重複人物がないか

ベンチマークはmacOSでは生成中の空きメモリ率も記録する。結果はプロファイルの品質だけでなく、メモリの最低値と生成時間も含めて評価する。

## 8. 出力CSV仕様

既存11列の順序を維持し、末尾に`image_path`と`image_seed`を追加する。

```text
name, profile, catchphrase, image_prompt, concept,
age, gender, species, ability, wants, role,
image_path, image_seed
```

画像を生成しない場合、`image_path`と`image_seed`は空欄にする。

## 9. エラー処理

1キャラクターの全生成（LLMと画像生成）が完了するまで、output.csvやseed CSVへ書き込まない。

途中で失敗した場合は、現在のrunを中断し、完了済みの行だけを保持する。run直下の`errors.jsonl`へ次のJSONを1行追加する。

```json
{"iteration":2,"stage":"profile","error_type":"TimeoutError","message":"inference timed out","timestamp":"2026-08-27T12:34:56Z"}
```

`stage`は、`character_concept`、`name`、`profile`、`catchphrase`、`new_ability`、`new_wants`、`new_role`、`image`、`persistence`、`input`のいずれかとする。

失敗時の終了コードは1とし、既存のrunディレクトリは削除しない。

## 10. 実行例

```bash
# ローカルLLMのみ
python ollama_hero_gen.py --iterations 1

# Ollama + ComfyUIでテキストと画像を生成
python ollama_hero_gen.py --iterations 1 --generate-images

# 任意のクラウドLLM + ローカルComfyUI画像生成
python ollama_hero_gen.py --provider openai --iterations 1 --generate-images
```

## 11. テスト方針

- 通常の`pytest -q`は外部ネットワーク、Ollama実サーバー、ComfyUI実サーバー、OpenAI APIを使用しない
- Ollama、ComfyUI、OpenAIはモックする
- 実サービスを使うテストは`integration`または`slow` markerで分離する
