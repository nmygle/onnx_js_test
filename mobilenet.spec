# -*- mode: python -*-

block_cipher = None


a = Analysis(['mobilenet.py'],
             pathex=['/home/nao/workspace/eel/onnx_js_test'],
             binaries=[],
             datas=[('/home/nao/.local/share/virtualenvs/eel-tKfCVlBM/lib/python3.7/site-packages/eel/eel.js', 'eel'), ('web', 'web')],
             hiddenimports=['bottle_websocket'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='mobilenet',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=False )
