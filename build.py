from flask import Flask
from flask_assets import Bundle, Environment

app = Flask(__name__)

# Bundling src/main.css files into dist/main.css'
css = Bundle('src/main.css', output='dist/main.css', filters='postcss')

assets = Environment(app)
assets.register('main_css', css)
css.build()