"""Exercise the URL gate with only stdlib, independently of CUDA packages."""
import ast
import ipaddress
import unittest
from pathlib import Path
from urllib.parse import urlsplit

tree = ast.parse(Path('app/worker.py').read_text())
function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'validate_url')
namespace = {'ipaddress': ipaddress, 'urlsplit': urlsplit}
exec(compile(ast.Module(body=[function], type_ignores=[]), '<url-validator>', 'exec'), namespace)
validate = namespace['validate_url']

class URLSecurityTests(unittest.TestCase):
    def test_private_and_metadata_addresses(self):
        for url in ['http://127.0.0.1/x', 'http://10.0.0.1/x', 'http://169.254.169.254/',
                    'http://[::1]/', 'http://[::ffff:127.0.0.1]/', 'http://192.168.1.1/']:
            with self.subTest(url=url), self.assertRaises(ValueError):
                validate(url)

    def test_scheme_and_credentials(self):
        for url in ['file:///etc/passwd', 'ftp://example.com/a', 'http://u:p@example.com/a', 'http:///x']:
            with self.subTest(url=url), self.assertRaises(ValueError):
                validate(url)

    def test_public_inputs(self):
        for url in ['https://example.com/a.png', 'https://8.8.8.8/a.png', 'https://[2606:4700:4700::1111]/a']:
            validate(url)

if __name__ == '__main__':
    unittest.main()
