pyinstaller -F ^
--distpath pyinstaller\dist ^
--name demthep_infer ^
--workpath pyinstaller\build ^
--add-binary=d:\miniconda3\envs\yolox-build\Lib\site-packages\onnxruntime\capi\onnxruntime_providers_shared.dll;./onnxruntime/capi/ ^
interfaces\async_server_inference.py