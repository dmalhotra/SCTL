# Runs the Makefile's tests, also after a failure, each failure as a GitHub annotation; exits with 1 if any failed.
# SCTL_TESTS: the tests to run instead; SCTL_TEST_RUNNER: a command to run them under
set +e
s=0
for t in ${SCTL_TESTS:-$(sed -n '/^test: /,/^$/p' Makefile | grep -o '/test[^/ ]*$' | cut -c2-)}; do
  echo "=== $t"
  $SCTL_TEST_RUNNER ./bin/$t 2>&1 | tee $t.log
  st=${PIPESTATUS[0]}
  echo "=== $t: exit $st"
  if [ $st -ne 0 ]; then
    s=1
    msg=$(grep -a -m4 -E 'Assertion|failed|Error|error' $t.log | cut -c1-500 | awk '{printf "%s%%0A", $0}')
    [ -n "$msg" ] || msg=$(tail -n 5 $t.log | cut -c1-500 | awk '{printf "%s%%0A", $0}')
    echo "::error title=$t exit $st::$msg"
  fi
done
exit $s
