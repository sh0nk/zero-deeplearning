#!/bin/bash
coverage run -m unittest -v test.all_suite
echo "=== COVERAGE REPORT ==="
coverage report
echo "=== OUTPUT TEST DETAIL REPORT ==="
coverage html
echo "output report in src/cover/"
