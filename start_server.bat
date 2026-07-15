@echo off
start /B "" "E:\my apps\LLAMA\llama-server.exe" -m "E:\my apps\LLAMA\gemma-4-E2B-it-Q4_K_S.gguf" -ngl 99 --port 8080 --ctx-size 8192
