# 使い方

このドキュメントは、実際にコードを動かして確認したことだけを書く。動かしていない主張は「未検証」と明記する。

## 進捗

| 段 | 関数 | 状態 |
|---|---|---|
| 第1段（水系を描く） | `experiments.mini.gen_headward` | 動作確認済み（2026-09-27） |
| 第2段（面を起こす） | `experiments.mini.solve` | 動作確認済み・既知の問題を部分修正（2026-09-27。下記「既知の問題」参照） |
| 第3段（仕上げ） | `experiments.mini_finish.finish_micro` | 未検証 |
| 正準の入口 | `experiments.baseline.generate` | 未検証（凍結ゼロの再現に問題あり。下記「環境と既知の問題」参照） |

## 環境

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

このリポジトリで実際に確認した組み合わせ:

- Python 3.14.7、numpy 2.5.3、scipy 1.18.1、scikit-image 0.26.0、matplotlib 3.11.2

`experiments/baseline.py` のコメントが指定する、開発時に検証された組み合わせ:

- Python 3.11.15、numpy 2.4.6、scipy 1.17.1、scikit-image 0.26.0、matplotlib 3.11.2

### 環境と既知の問題

`generate()` の「何度呼んでもビット単位で同じものを返す」という設計は、次のコマンドで検証できる。

```bash
python -m experiments.baseline --check
```

開発時に検証された組み合わせ（上記）では `max |u - golden| = 0.00e+00` が確認されている。

**この環境（numpy 2.5.3 / scipy 1.18.1 / Python 3.14.7）で実行すると、`max |u - golden| = 5.28e-02` となり、一致しない。** 原因は未調査。`scipy.ndimage.distance_transform_edt` など、内部で使う数値計算ルーチンのバージョン差が候補として考えられるが、切り分けていない。

`generate()` を使う際は、この既知の問題を踏まえること。第1段（`gen_headward`）単体は、この環境でも実行でき、後述のとおり実用上問題のない結果が得られている。

---

## 図の作り方

以下の図はすべて `matplotlib` で作った。ユーザーが自分のデータを見るときの参考として、使った関数をそのまま載せる。

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# 第1段(gen_headward)の出力: bool 配列 net、int16 配列 lab を色分けする
def plot_net(net, lab, title):
    if lab is None:
        rgb = np.where(net[..., None], np.array([0x2a, 0x78, 0xd6]) / 255.0,
                       np.array([0xfc, 0xfc, 0xfb]) / 255.0)
    else:
        colors = {0: "#fcfcfb", 1: "#2a78d6", 2: "#eb6834"}  # land, basin 1, basin 2
        rgb = np.zeros(net.shape + (3,))
        for k, hexcode in colors.items():
            rgb[lab == k] = np.array([int(hexcode[i:i+2], 16) for i in (1, 3, 5)]) / 255.0
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.imshow(rgb, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title)
    fig.savefig("net.png", facecolor="#fcfcfb")

# 第2段(solve)の出力: float64 配列 u(標高)を連続カラーマップで表示し、川を重ねる
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
])

def plot_elev(u, net, title):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(u, cmap=SEQ_BLUE, interpolation="nearest")
    ry, rx = np.where(net)
    ax.scatter(rx, ry, s=1.5, c="#0b0b0b", alpha=0.55, linewidths=0)  # 川を重ねる
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="elevation (0-1, normalized)")
    fig.savefig("elev.png", facecolor="#fcfcfb")
```

配色は `dataviz` スキル（バリデータ `validate_palette.js`）で検証済みのもの: 流域の色分け（青 `#2a78d6` / 橙 `#eb6834`）は色覚多様性のもとでも判別できる組み合わせとして確認した。標高は単一色（青）の連続カラーマップにしている。

---

## 第1段 — `gen_headward`（水系を描く）

```python
def gen_headward(rng, n=128, ell=5.0, ..., roots=None, shape=None,
                 labels=False, domain=None, ...)
```

必須は乱数生成器 `rng` だけ。残りは全部省略できる。

主な引数:

| 引数 | 意味 |
|---|---|
| `n` | 正方形の一辺（px）。`shape` を指定すると無視される |
| `shape=(H, W)` | 長方形にしたい場合の大きさ |
| `roots` | `(行, 列, 向き)` のリスト。水の出口（省略時は下辺中央に1つ） |
| `domain` | bool 配列。True が陸。省略すると長方形全面が陸になる |
| `reserve` | bool 配列。川を生やしてはいけない場所 |
| `labels` | **常に `True` を渡すこと。** 第2段 `solve` に渡す `lab` を得るために必要 |

戻り値（`labels=True` で呼んだ場合）:

`(net, lab)` のタプル。`net` は bool 配列で、川なら `True`。`lab` は int16 配列で、`-1` = 陸の外、`0` = 陸だが川ではない、`k > 0` = k 番目の出口に注ぐ川。

標高はこの関数の入力に無い。地形を見て川を引いているのではなく、白紙に川を引いている。

### 例1：出口1つ

```python
import numpy as np
from experiments.mini import gen_headward

net, lab = gen_headward(np.random.default_rng(3), n=36, labels=True)
```

![Stage 1, one outlet: a single branching blue channel network on a light background, growing from the bottom edge.](images/stage1_single.png)

### 例2：出口を2つ置き、流域も返させる

```python
roots = [(35, 12, -np.pi/2),   # 下辺  → 上へ遡る
         (18, 35,  np.pi)]     # 右辺  → 左へ遡る
net, lab = gen_headward(np.random.default_rng(5), n=36, roots=roots, labels=True)
```

`roots` は `(行, 列, 向き)`。向きは水が流れる方ではなく、木が伸びる方＝上流方向。`lab` の実際の値は `{0, 1, 2}`。

![Stage 1, two outlets: two separate branching networks, one blue and one orange, that never touch each other.](images/stage1_two.png)

流域1（青）と流域2（橙）が混ざらず、あいだに何も無い帯（＝分水界）ができている。この分け方を `lab` として第2段に渡す。

**注記。** この結果は、同じシード・同じ引数で以前得られていた記録とほぼ同じ地形の骨格になったが、数ピクセル分だけ経路が異なった。前述の環境差（numpy/scipy のバージョン）が原因と見ているが、確認していない。

---

## 第2段 — `solve`（面を起こす）

```python
def solve(net, tilt=None, poisson=0.006, theta=0.3, cap_px=0.0, floor_px=4.0,
          outlets=None, acc_ref=None, bow=0.0, floor_mul=1.0, cone=True,
          floor_tilt=0.0, hollow_px=0.0, labels=None)
```

第1段が返す `net`（1px の川のラスタ）を受け取り、標高の面を返す。

主な引数:

| 引数 | 意味 |
|---|---|
| `net` | 第1段の出力（bool 配列） |
| `outlets` | `(行, 列)` のリスト。第1段で使った `roots` の座標をそのまま渡す |
| `labels` | **常に渡すこと。** 第1段から `labels=True` で受け取った `lab` をそのまま渡す |
| `theta` | 川の縦断形状の凹み具合（Flint の法則の指数） |
| `poisson` | 丘を下から押し上げる強さ。0 だと谷の間の面は単なる調和補間（膜を張っただけ）になり、丘が川の高さを超えられない |
| `floor_px` | 谷底の幅 |

戻り値は `float64` の2次元配列（`net` と同じ形）。値は `[0, 1]` に正規化されている。`0` が最も低い（出口の高さ）。

### 既知の問題：川が谷底にない（2026-09-27, 部分修正）

最初に作った版では、川セル（`net==True`）の標高が、隣接する陸セルより**高い**ことが大半だった（n=128 で87.8〜90.2%、8近傍のうち自分が最下位＝正しい谷底になっているのはわずか0.1〜0.2%）。原因は、`solve()` が縦断標高を「1px の川」ではなく「川を1px 太らせた帯」の上で計算していたこと。帯の脇のセルは、曲がり角で本来の川の経路をショートカットでき、集水量も別物になるため、勾配の計算が本来の川とずれる。

**修正内容。** 縦断標高（測地距離・集水量・Flint則の積分）を1px の `net`（＋頭部延長）だけで計算し、帯の脇や谷底のセルは、最も近い川セルの標高をそのまま塗るように変更した。谷底(`floor_px`)の塗り方はもともとこの方式だったので、帯の脇にも同じ考え方を広げた形になる。

**修正後の実測（このリポジトリの検査で確認できた範囲）:**

| | 修正前 | 修正後 |
|---|---|---|
| n=128, 出口1つ | 87.8% | 79.0% |
| n=128, 出口2つ | 90.2% | 79.6% |
| n=36, 出口2つ | 83.0% | 74.0% |

（「川セルが、隣接する非河川セルの最低値より高い」割合。8近傍・自作の簡易な検査）

**まだ直っていない。** 数値は改善したが、「ほぼ0%」には程遠い。内訳を調べると、残った違反のうち約6割は、帯の脇のセルが「直線距離で一番近い川セル」を持ち主としているため、実際には隣接していない・標高がまったく違う川の別の部分（急カーブの先など）に所有権が飛んでしまうケースだった。直線距離ではなく、川に沿った距離（測地距離）で持ち主を決めれば、さらに改善する可能性がある。残り約4割は、川セル同士が raster 上で隣り合っているのに、川に沿った距離では離れている（ヘアピン状の経路）ことによる、より根本的な限界かもしれない。

この修正は、別の開発環境での調査結果をもとに実装した。開発環境からは「実物と同じ水準（3%程度）まで改善する」と報告されたが、このリポジトリで再現した数値はそれよりかなり高い。測っている指標が違う可能性がある（開発環境の指標の詳細は分かっていない）。

### 例1：出口1つ（第1段の続き）

```python
from experiments.mini import gen_headward, solve
import numpy as np

net, lab = gen_headward(np.random.default_rng(3), n=36, labels=True)
u = solve(net, outlets=[(34, 18)], labels=lab)   # n=36 の既定の出口は (n-2, n//2) = (34, 18)
```

![Stage 2, one outlet: a smooth blue elevation surface, lightest near the outlet at the bottom and darkest at the far edges, with small dark dots marking the channel.](images/stage2_single.png)

実際の値（`u.min()` = 0.0、`u.max()` = 0.9999999999991916、NaN 無し）。出口 `(34, 18)` の周辺が最も低く（明るい色）、地図の縁に近づくほど高くなる（濃い色）。川が1本しかない場合は、分水界（等距離の帯）が生まれる場所が無く、面全体がひとつの丘としてなだらかに盛り上がる。

### 例2：出口を2つ

```python
from experiments.mini import gen_headward, solve
import numpy as np

roots = [(35, 12, -np.pi/2),
         (18, 35,  np.pi)]
net, lab = gen_headward(np.random.default_rng(5), n=36, roots=roots, labels=True)
u = solve(net, outlets=[(r, c) for (r, c, _a) in roots], labels=lab)
```

![Stage 2, two outlets: a blue elevation surface with two light troughs following the two channels, and a smooth ridge rising exactly halfway between them.](images/stage2_two.png)

実際の値（`u.min()` = 0.0、`u.max()` = 0.9999999999990007、NaN 無し）。出口の座標（`(35,12)` と `(18,35)`）の周辺で値が最も低くなっており、外側に向かって高くなっている。色の変化はなめらかで、丘の頂上が個々の川から等距離の場所に自然に現れている（ドラフトにあった「尾根は描いていない」という主張どおりの挙動）。

---

## 第3段

未検証。次の作業でここに追記する。
