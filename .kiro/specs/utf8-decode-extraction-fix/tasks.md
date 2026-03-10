# Implementation Plan

- [x] 1. Write bug condition exploration test
  - **Property 1: Fault Condition** - UTF-8 Decode Error Handling
  - **CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior - it will validate the fix when it passes after implementation
  - **GOAL**: Surface counterexamples that demonstrate the bug exists (UnicodeDecodeError crashes)
  - **Scoped PBT Approach**: Scope the property to concrete failing cases - decryption returns invalid UTF-8 bytes
  - Test that extract_message_ui crashes with UnicodeDecodeError when decryption produces invalid UTF-8 bytes (from Fault Condition in design)
  - Mock encryption_service.decrypt() to return invalid UTF-8 bytes: `b'\x9b\xf2\xa3\xc4'`, `b'\xff\xfe\xfd\xfc'`, `b'Hello\x9b\xf2'`
  - The test assertions should match the Expected Behavior Properties from design: catch exception and return user-friendly error message
  - Run test on UNFIXED code
  - **EXPECTED OUTCOME**: Test FAILS with UnicodeDecodeError propagating uncaught (this is correct - it proves the bug exists)
  - Document counterexamples found: specific error messages like `UnicodeDecodeError: 'utf-8' codec can't decode byte 0x9b in position 0: invalid start byte`
  - Mark task complete when test is written, run, and failure is documented
  - _Requirements: 2.1, 2.2, 2.3_

- [x] 2. Write preservation property tests (BEFORE implementing fix)
  - **Property 2: Preservation** - Successful Extraction Behavior
  - **IMPORTANT**: Follow observation-first methodology
  - Observe behavior on UNFIXED code for non-buggy inputs (valid UTF-8 messages, ExtractionError cases, validation errors)
  - Write property-based tests capturing observed behavior patterns from Preservation Requirements:
    - Successful extraction with valid UTF-8 messages decodes and displays correctly
    - ExtractionError exceptions are caught and handled with existing error message format
    - Password validation errors are returned before attempting extraction
    - Image validation errors are returned before attempting extraction
  - Property-based testing generates many test cases for stronger guarantees (random valid UTF-8 messages, various passwords, different image scenarios)
  - Run tests on UNFIXED code
  - **EXPECTED OUTCOME**: Tests PASS (this confirms baseline behavior to preserve)
  - Mark task complete when tests are written, run, and passing on unfixed code
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [x] 3. Fix for UTF-8 decode error in extraction

  - [x] 3.1 Implement the fix
    - Add UnicodeDecodeError handler after the existing ExtractionError handler in pixelnur/app.py (after line 265)
    - Add except clause: `except UnicodeDecodeError as e:`
    - Log the decode failure: `logger.warning(f"UTF-8 decode error: {str(e)}")`
    - Return user-friendly error message with ❌ emoji, explaining possible causes (wrong password, modified image, no embedded data) and actionable guidance
    - Maintain consistent error format with existing error messages (bullet points, helpful tone)
    - Preserve existing exception order (ExtractionError first, UnicodeDecodeError second)
    - _Bug_Condition: isBugCondition(input) where extraction_and_decryption_complete AND NOT is_valid_utf8(message_bytes) AND decode_attempted_
    - _Expected_Behavior: Catch UnicodeDecodeError and return user-friendly error message with actionable guidance (from Property 1 in design)_
    - _Preservation: All successful extraction behavior, ExtractionError handling, validation error handling unchanged (from Preservation Requirements in design)_
    - _Requirements: 2.1, 2.2, 2.3, 3.1, 3.2, 3.3, 3.4_

  - [x] 3.2 Verify bug condition exploration test now passes
    - **Property 1: Expected Behavior** - UTF-8 Decode Error Handling
    - **IMPORTANT**: Re-run the SAME test from task 1 - do NOT write a new test
    - The test from task 1 encodes the expected behavior (catch exception, return user-friendly error)
    - When this test passes, it confirms the expected behavior is satisfied
    - Run bug condition exploration test from step 1
    - **EXPECTED OUTCOME**: Test PASSES (confirms bug is fixed - no more crashes, returns error messages instead)
    - _Requirements: 2.1, 2.2, 2.3_

  - [x] 3.3 Verify preservation tests still pass
    - **Property 2: Preservation** - Successful Extraction Behavior
    - **IMPORTANT**: Re-run the SAME tests from task 2 - do NOT write new tests
    - Run preservation property tests from step 2
    - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions in successful extraction, ExtractionError handling, validation)
    - Confirm all tests still pass after fix (no regressions)

- [ ] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
