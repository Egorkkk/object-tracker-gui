"""Local HTTP transport. GPU operations run exclusively in Application's worker."""
import argparse
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import mimetypes
from pathlib import Path
import shutil
from urllib.parse import parse_qs, urlparse
import cv2
import numpy as np

from object_tracker.app.service import Application
from object_tracker.backends.gotrack.backend import GoTrackBackend
from object_tracker.core.project import Project
from object_tracker.core.storage import local_path
from object_tracker.core.viewer import render

ROOT = Path(__file__).resolve().parents[3]


def handler_class(application):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, message, *args):
            logging.getLogger('http').debug(message, *args)

        def respond(self, value, status=200, content_type='application/json'):
            data = json.dumps(value, allow_nan=False).encode() if content_type == 'application/json' else value
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            try:
                parsed = urlparse(self.path)
                query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
                if parsed.path == '/api/state':
                    return self.respond(application.snapshot())
                if parsed.path == '/api/browse':
                    path = local_path(query.get('path') or Path.home())
                    if path.is_file():
                        path = path.parent
                    entries = []
                    for entry in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                        if entry.name.startswith('.'):
                            continue
                        entries.append(dict(name=entry.name, path=str(entry), directory=entry.is_dir()))
                    return self.respond(dict(path=str(path), parent=str(path.parent), entries=entries))
                if parsed.path == '/api/environment':
                    return self.respond(dict(python=__import__('sys').executable,
                        checkpoint=(ROOT/'external/gotrack/gotrack_checkpoint.pt').is_file(),
                        dino_weights=(Path.home()/'.cache/torch/hub/checkpoints/dinov2_vits14_reg4_pretrain.pth').is_file(),
                        ffmpeg=shutil.which('ffmpeg'), ffprobe=shutil.which('ffprobe'),
                        gpu_probe='CUDA/EGL проверяются worker перед загрузкой модели'))
                if parsed.path == '/api/frame':
                    with application.lock:
                        project = application.required()
                        index = project.check_index(query.get('index', 0))
                        snapshot = Project(project.directory, deepcopy(project.state))
                        snapshot.sequence = project.sequence
                        mesh = application.mesh
                    image = render(snapshot, mesh, index, initial=query.get('initial', '1') == '1',
                                   refined=query.get('refined', '1') == '1', masks=query.get('masks', '1') == '1',
                                   object_mask=query.get('object_mask', '1') == '1', occlusion_mask=query.get('occlusion_mask', '1') == '1',
                                   mode=query.get('mode', 'contour'), opacity=max(0., min(1., float(query.get('opacity', .5)))))
                    ok, data = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    if not ok:
                        raise ValueError('Не удалось сформировать preview кадра')
                    return self.respond(data.tobytes(), content_type='image/jpeg')
                if parsed.path == '/api/download':
                    with application.lock:
                        p = application.required()
                        target = (p.directory / query['path']).resolve()
                        allowed = [p.directory / name for name in ('exports', 'previews')]
                    if not any(target.is_relative_to(folder.resolve()) for folder in allowed) or not target.is_file():
                        raise ValueError('Недопустимый путь экспорта')
                    self.send_response(200)
                    self.send_header('Content-Type', mimetypes.guess_type(target.name)[0] or 'application/octet-stream')
                    self.send_header('Content-Length', str(target.stat().st_size))
                    self.send_header('Content-Disposition', 'attachment; filename="' + target.name + '"')
                    self.end_headers()
                    with target.open('rb') as stream:
                        shutil.copyfileobj(stream, self.wfile)
                    return
                pages = {'/': 'index.html', '/app.js': 'app.js', '/style.css': 'style.css'}
                if parsed.path not in pages:
                    return self.respond(dict(error='Не найдено'), 404)
                path = ROOT / 'frontend' / pages[parsed.path]
                return self.respond(path.read_bytes(), content_type=(mimetypes.guess_type(str(path))[0] or 'text/plain') + '; charset=utf-8')
            except Exception as exc:
                logging.getLogger(__name__).exception('GET failed')
                self.respond(dict(error='Ошибка: ' + str(exc)), 400)

        def do_POST(self):
            try:
                # Local file access must not be triggered by a foreign web origin.
                origin = self.headers.get('Origin')
                if origin and urlparse(origin).netloc != self.headers.get('Host'):
                    return self.respond(dict(error='Запрос с другого origin запрещён'), 403)
                if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                    return self.respond(dict(error='Требуется application/json'), 415)
                size = int(self.headers.get('Content-Length', 0))
                if size > 2_000_000:
                    raise ValueError('Слишком большой запрос')
                body = json.loads(self.rfile.read(size))
                path = urlparse(self.path).path
                if path == '/api/create': result = application.load(body, True)
                elif path == '/api/open': result = application.load(body)
                elif path == '/api/pose': result = application.change_pose(body)
                elif path == '/api/pose/import':
                    result = application.change_pose(dict(index=body['index'], matrix=np.load(local_path(body['path']), allow_pickle=False).tolist()))
                elif path == '/api/refine': result = application.refine(body['index'])
                elif path == '/api/accept': result = application.accept(body['index'], True)
                elif path == '/api/reject': result = application.accept(body['index'], False)
                elif path == '/api/track': result = application.track(body)
                elif path == '/api/resume': result = application.track(body, True)
                elif path == '/api/cancel': result = application.stop()
                elif path == '/api/settings': result = application.settings(body)
                elif path == '/api/masks': result = application.masks(body)
                elif path == '/api/export': result = application.export()
                elif path == '/api/preview': result = application.preview()
                elif path == '/api/save':
                    with application.lock:
                        application.required().save()
                    result = application.snapshot()
                elif path == '/api/frame/select':
                    with application.lock:
                        p = application.required()
                        p.state['current_frame'] = p.check_index(body['index'])
                        p.save()
                    result = application.snapshot()
                else: return self.respond(dict(error='Не найдено'), 404)
                self.respond(result)
            except Exception as exc:
                logging.getLogger(__name__).exception('POST failed')
                self.respond(dict(error='Ошибка: ' + str(exc)), 400)
    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--project')
    parser.add_argument('--gotrack', type=Path, default=ROOT / 'external/gotrack')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'external/gotrack/gotrack_checkpoint.pt')
    args = parser.parse_args()
    logs = ROOT / 'outputs/logs'; logs.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s',
                        handlers=[logging.FileHandler(logs/'application.log'), logging.StreamHandler()])
    handler = logging.FileHandler(logs / 'backend_gotrack.log')
    handler.addFilter(lambda record: record.name.startswith(('model.', 'utils.', 'object_tracker.backends.')))
    logging.getLogger().addHandler(handler)
    app = Application(lambda output: GoTrackBackend(args.gotrack, args.checkpoint, output, ROOT / 'outputs/runtime'))
    if args.project:
        app.load(dict(directory=args.project))
    server = ThreadingHTTPServer((args.host, args.port), handler_class(app))
    print(f'Откройте http://{args.host}:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        app.close()


if __name__ == '__main__':
    main()
