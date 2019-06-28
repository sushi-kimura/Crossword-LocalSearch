"""
* 特定のディレクトリ内の画像を動画に変換するプログラム
* 入力画像ファイルの拡張子は設定可能
* 以下のパラメータを設定して実行する
  imagedir: 入力画像の入っているディレクトリのパス
  input_ext: 入力画像の拡張子
  codec: 動画コーデックの設定は http://www.fourcc.org/codecs.php を参照
  Output: codecに対応した拡張子を持つ出力ファイル名
  fps: 出力する動画の1秒あたりのフレーム数
"""
import glob
import cv2

# Settings
imagedir = "output/optflow_vorticity/"
input_ext = "png"
codec = ["m", "p", "4", "v"]
output = "out.mov"
fps = 12

# Open images
files = sorted(glob.glob(imagedir+'/*.'+ input_ext))
images = list(map(lambda file: cv2.imread(file), files))

fourcc = cv2.VideoWriter_fourcc(codec[0], codec[1], codec[2], codec[3])
video = cv2.VideoWriter(output, fourcc, fps, (images[0].shape[0], images[0].shape[1]))

# Sage as movie
for img in images:
    video.write(img)

video.release()
