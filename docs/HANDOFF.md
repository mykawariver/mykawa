# 作業の引き継ぎメモ

最終更新：2026-10-25。中断したところから再開するためのメモです。

## いまの状態

### 基準の地形（`experiments/baseline.py` の `generate()`）

- 調整値をすべて 0 にした地形が基準です。第1段は `gen_headward` を使います。
- 流出点は `edge_roots()` で、地図の外周の行・列そのものに置きます（3〜5個、四辺の12〜88%の範囲）。
- `solve` には `acc_ref` と `hollow_px` を渡しません（既定値の `None` と `0`）。
- 基準のデータ：`valley_samples/cache/baseline_zero_s1.npz`
  - `python -m experiments.baseline --check` で差が `0.00e+00` になることを確かめる
  - 確認した環境：Python 3.13.16 / numpy 2.5.3 / scipy 1.18.1 / scikit-image 0.26.0 / matplotlib 3.11.2
- `generate(stage1="flood")` で、第1段を `gen_flood` にできます。既定は `"headward"` です。

### 第1段の2つの関数

| | `mini.gen_headward` | `flood.gen_flood` |
|---|---|---|
| 方式 | 川の頭を、排水されていない土地へ伸ばしていく | ランダムな起伏の上で、流出点から水を広げる（最小全域木）。集水量のしきい値で川にする |
| 128×128 | 約1秒 | 約0.1秒 |
| 256×1024（3 km × 12 km） | 第1段だけで約170秒 | 約3.5秒 |
| 調整値 | `ell`（density）、`floor_px`（floor）、流出点（roots） | 同じ |
| `trend`（流れの向きの制約） | あり（検証は不十分） | なし |
| 性質 | 支流が細かく均一。線は短い線のつながり | 川の頭が少なめで区間が長め。1ピクセル単位でくねる |

- `gen_headward` は、既定の流出点を下辺の上 `(H-1, W//2)` に置きます（以前は1ピクセル内側でした）。
- `gen_headward` は、最後の細線化のあとで、流出点と川のつながりを保ちます（`mini.rejoin_after_thinning`）。
- `experiments/headward2.py`（`gen_headward2`）は、高速化を試みた途中の版です。結果は `gen_headward` とビット単位で同じですが、2倍程度しか速くなっていません。どこからも使っていません。

### 文書

- **README.md と README.ja.md：** 3段階の例（36×36）を、画像つきで載せています。
- **docs/USAGE.md：** 「この下は編集中です！！！」より上を推敲済みです。第1段〜第3段に、それぞれ例1〜例4があります。
  - 例1：流出点1つ
  - 例2：流出点2つ
  - 例3：128×128（`generate(1)` と同じ）
  - 例4：東に海
- **docs/images/：**
  - USAGE.md の図16枚
  - その `gen_flood` 版16枚（`*_flood.png`）
  - 3 km × 12 km の比較図2枚（`headward_3x12km_s1.png`、`flood_3x12km_s1.png`）

## 道具（`tools/`）

| コマンド | 内容 |
|---|---|
| `python -m tools.usage_figures docs/images` | USAGE.md の図16枚を描き直す |
| `python -m tools.usage_figures docs/images --stage1 flood` | 同じく `gen_flood` 版（`*_flood.png`。既存の図は上書きしない） |
| `python -m tools.knob_sweep --stage1 flood` | 6つの調整値 × 5段階の等高線の一覧図（30枚） |
| `python -m tools.long_map --stage1 flood` | 3 km × 12 km の図 |
| `tools.network_metrics.sweep("flood", floor=2)` | 調整値ごとの川の網の数字（シード1〜5の平均）。説明に、2つの関数の参考値の表があります |

`usage_figures` は、今リポジトリにある32枚を、ビット単位で再現することを確かめてあります。

## 宿題

1. **大きな地図での `gen_flood` の網の薄さ**
   - 3 km × 12 km で、`gen_flood` の網は `gen_headward` より薄くなります（川の密度 4.78% に対して 6.22%、川の頭 308 に対して 694、最大の空白 30 に対して 15 ピクセル）。128×128 ではほぼ同じです。
   - 谷底の処理（`FLOOR_REACH = 2.0` で谷底を支流の集水域から外す処理と、谷底に入った小さな支流を取り除く処理）が、長い本流の太い谷底で効きすぎているのだと思います。
   - 地図の大きさに合わせて見直す必要があります。`tools/long_map.py` と `tools/network_metrics.py` で比べられます。
2. **`gen_headward` の `trend` の検証**
   - 64×256 で、逆向きの流れが 29.7% から 2.2% に減ることだけ確かめました。
   - 尾根が途切れないか、`trend_max` の効き、乾いた土地が残りすぎないか、を確かめる必要があります。
3. **USAGE.md の「編集中」より下の古い記述**
   - 「`--check` が一致しない」（今は一致します）、`(34, 18)`、以前の数値などが残っています。
4. **既定の流出点を変えた影響**
   - 実験用コード（`experiments/mini.py` の評価、`experiments/mini_finish.py:229`）の結果が変わっている可能性があります。
5. **調整値の効きの統計的な比較**
   - `gen_flood` と `gen_headward` をシード30個ほどで比べます。今はシード1〜5と、シード1の一覧図だけです。
6. **基準の地形を `gen_flood` にするかどうか**
   - 変えるなら、基準のデータを作り直します（`--freeze`）。
7. **`gen_flood` に `trend` に相当する機能を足すかどうか**

## 作業の約束事

- 作業ブランチで作業し、確かめてから `main` に fast-forward でマージします。
- 基準の地形を変えたら、`--freeze` で作り直し、`--check` で差 0 を確かめます。
- 図を新しく作るときは、既存のファイルを上書きしない名前にします。
- USAGE.md の例のコードは、そのまま実行して動くことを確かめます。
