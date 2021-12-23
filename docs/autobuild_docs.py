#!/usr/bin/env python
from livereload import Server, shell
server = Server()
server.watch('../tomopyui/widgets/main.py', shell('sphinx-build . _build -b html'))

server.serve(root='_build/html')