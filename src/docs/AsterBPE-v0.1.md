# AsterBPE-v0.1

## 入力:
    * Python str

## 基礎語彙:
    * token ID 0..255
        = UTF-8の1 byteと完全一致

## 学習:
    1.CorpusをUTF-8 bytesへ変換
        ↓
    2.隣接token pairを数える
        ↓
    3.最頻pairを1 tokenへmerge
        ↓
    4.新token IDを256, 257, 258...と順番に付与
        ↓
    5.target_vocab_sizeまで反復

## 利用:
    1.str
        ↓ UTF-8
    2.byte IDs
        ↓ learned merge rules
    3.BPE token IDs

## 復号:
    1.token IDs
        ↓
    2.各tokenが表すbytesを連結
        ↓ UTF-8
    3.str