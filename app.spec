# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['app.py'],
             pathex=['C:\\Users\\jasnell\\Documents\\Github\\game-disruptor-2'],
             binaries=[],
             datas=[
                ('templates', 'templates'),
                ('static', 'static'),
             ],
             hiddenimports=['clr'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
splash = Splash('boot_splash.png',
                binaries=a.binaries,
                datas=a.datas,
                text_pos=None,
                text_size=12,
                minify_script=True)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          splash,
          splash.binaries,
          [],
          name='GameDisruptor',
          debug=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=True)

