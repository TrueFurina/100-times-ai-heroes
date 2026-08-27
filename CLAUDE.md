# Claude Code Instructions

## プロジェクト概要
100 Times AI Heroes - Ollama版キャラクター生成システム

## セッション開始時の必須タスク

**毎セッション開始時に以下を実行すること:**

1. `cat harness/claude-progress.txt` で前回の進捗を確認
2. `python -c "import json; d=json.load(open('harness/features.json')); print(f'{sum(1 for f in d[\"features\"] if f[\"passes\"])}/{len(d[\"features\"])} features passing')"` で機能進捗確認
3. `git log --oneline -3` で最近のコミット確認
4. 次に実装すべき機能を `features.json` から選定

## 開発ルール

### 一度に1機能のみ
- features.json から `"passes": false` の機能を1つ選ぶ
- その機能のテストをパスさせることだけに集中
- パスしたら次の機能へ

### テスト駆動
```bash
# テスト実行
pytest harness/test_harness.py -v -k "<category>"

# 機能リスト自動更新
python harness/test_harness.py
```

### コミット粒度
- 1機能完了 = 1コミット
- コミットメッセージ: `feat: <機能ID> <説明>`

## ファイル構成

```
ollama_hero_gen.py      # 生成パイプライン、CSV、LLMプロバイダー
comfyui_image_gen.py    # localhost ComfyUI APIクライアント
image_model_profiles.py # 画像モデルprofileとprompt形式
memory_safety.py        # macOS/Linuxのメモリ安全弁
tools/benchmark_image_models.py # 候補モデル比較ベンチマーク
config/comfyui/         # ComfyUI API workflow
requirements-cloud.txt  # 任意のOpenAIプロバイダー用依存関係
harness/
├── features.json       # 機能リスト（passes を更新）
├── claude-progress.txt # 進捗ログ（セッション終了時に追記）
├── test_harness.py     # テストハーネス
├── init.sh            # 環境初期化
└── SESSION_PROTOCOL.md # 詳細プロトコル
```

## 技術スタック
- Python 3.9+
- Ollama (gpt-oss-20b)
- ローカルCSV
- ComfyUI（画像生成時のみ、localhost）
- OpenAI Responses API（任意、明示指定時のみ）
- M4 MacBook Pro (128GB)

## 設計仕様
詳細は `docs/DESIGN_SPEC_OLLAMA.md` を参照
