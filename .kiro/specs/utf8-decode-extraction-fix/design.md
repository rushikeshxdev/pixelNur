# UTF-8 Decode Extraction Fix Bugfix Design

## Overview

The Gradio app crashes with `UnicodeDecodeError` when extraction/decryption returns invalid UTF-8 bytes. This occurs at line 265 in `pixelnur/app.py` when calling `message_bytes.decode('utf-8')` on decrypted message bytes. The fix wraps the UTF-8 decode operation in proper error handling to catch `UnicodeDecodeError` and return a user-friendly error message that guides users to verify their password, check image integrity, and confirm the image contains embedded data. This ensures graceful degradation instead of application crashes while preserving all existing successful extraction behavior.

## Glossary

- **Bug_Condition (C)**: The condition that triggers the bug - when `message_bytes.decode('utf-8')` receives invalid UTF-8 bytes from decryption
- **Property (P)**: The desired behavior when invalid UTF-8 is encountered - catch the exception and return a user-friendly error message
- **Preservation**: Existing successful extraction behavior, ExtractionError handling, and validation error handling that must remain unchanged
- **extract_message_ui**: The function in `pixelnur/app.py` (around line 240-280) that handles message extraction from stego images in the Gradio interface
- **message_bytes**: The decrypted bytes returned by `encryption_service.decrypt()` that may or may not be valid UTF-8
- **UnicodeDecodeError**: Python exception raised when attempting to decode bytes that are not valid UTF-8

## Bug Details

### Fault Condition

The bug manifests when the decryption process produces bytes that are not valid UTF-8 encoding. This can occur due to an incorrect password (decryption produces garbage bytes), a corrupted or modified stego image (extraction returns invalid data), or extraction failure (returns random bytes). The `message_bytes.decode('utf-8')` call at line 265 raises an unhandled `UnicodeDecodeError`, causing the application to crash instead of providing user feedback.

**Formal Specification:**
```
FUNCTION isBugCondition(input)
  INPUT: input of type (message_bytes: bytes, password: str, stego_image: ndarray)
  OUTPUT: boolean
  
  RETURN extraction_and_decryption_complete(input.stego_image, input.password)
         AND NOT is_valid_utf8(input.message_bytes)
         AND decode_attempted(input.message_bytes)
END FUNCTION
```

### Examples

- **Wrong Password**: User provides incorrect password → decryption produces garbage bytes `b'\x9b\xf2\xa3...'` → `decode('utf-8')` raises `UnicodeDecodeError: 'utf-8' codec can't decode byte 0x9b in position 0: invalid start byte` → app crashes
- **Modified Image**: Stego image has been compressed or edited → extraction returns corrupted data → decryption produces invalid bytes → `decode('utf-8')` crashes
- **No Embedded Data**: User tries to extract from a regular image with no embedded message → extraction returns random pixel data → decryption produces non-UTF-8 bytes → crash occurs
- **Edge Case - Partial UTF-8**: Decryption produces bytes that are valid UTF-8 for the first N characters but invalid afterward → should decode successfully (no bug condition)

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- Successful extraction with valid UTF-8 messages must continue to decode and display correctly
- ExtractionError exceptions must continue to be caught and handled with the existing error message format
- Password validation errors must continue to be returned before attempting extraction
- Image validation errors must continue to be returned before attempting extraction
- Logging behavior for successful extractions must remain unchanged

**Scope:**
All inputs that do NOT involve invalid UTF-8 bytes from decryption should be completely unaffected by this fix. This includes:
- Valid password + valid stego image → successful extraction
- ExtractionError raised by extraction_engine.extract()
- Password validation failures caught by _validate_password()
- Image file reading failures (stego_bgr is None)

## Hypothesized Root Cause

Based on the bug description and code analysis, the root cause is clear:

1. **Missing Exception Handler**: The code at line 265 calls `message_bytes.decode('utf-8')` without wrapping it in a try-except block to catch `UnicodeDecodeError`
   - The existing try-except block only catches `ExtractionError`
   - `UnicodeDecodeError` is not a subclass of `ExtractionError`, so it propagates uncaught

2. **Assumption of Valid UTF-8**: The code assumes that decryption always produces valid UTF-8 bytes
   - This assumption breaks when the password is wrong (decryption produces garbage)
   - This assumption breaks when extraction fails silently (returns corrupted data)

3. **Insufficient Error Handling Scope**: The exception handling strategy doesn't account for decoding failures as a distinct error category from extraction failures

## Correctness Properties

Property 1: Fault Condition - UTF-8 Decode Error Handling

_For any_ input where decryption produces invalid UTF-8 bytes (isBugCondition returns true), the fixed extract_message_ui function SHALL catch the UnicodeDecodeError exception and return a user-friendly error message that explains possible causes (wrong password, modified image, no embedded data) and provides actionable guidance to the user.

**Validates: Requirements 2.1, 2.2, 2.3**

Property 2: Preservation - Successful Extraction Behavior

_For any_ input where decryption produces valid UTF-8 bytes (isBugCondition returns false), the fixed extract_message_ui function SHALL produce exactly the same behavior as the original function, successfully decoding and displaying the extracted message with the same format and logging.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4**

## Fix Implementation

### Changes Required

**File**: `pixelnur/app.py`

**Function**: `extract_message_ui` (around lines 240-280)

**Specific Changes**:
1. **Add UnicodeDecodeError Handler**: Add a new except clause after the existing ExtractionError handler to catch `UnicodeDecodeError`
   - Position: After line 265 where `message_bytes.decode('utf-8')` is called
   - Catch: `except UnicodeDecodeError as e:`

2. **Log the Decode Failure**: Add logging to record when UTF-8 decode fails for debugging purposes
   - Use: `logger.warning(f"UTF-8 decode error: {str(e)}")`
   - This helps diagnose whether the issue is password-related or data corruption

3. **Return User-Friendly Error Message**: Provide clear, actionable feedback to the user
   - Format: Similar to the existing ExtractionError message format
   - Content: Explain that decryption produced invalid data and list possible causes
   - Guidance: Suggest verifying password, checking image integrity, confirming embedded data exists

4. **Preserve Existing Exception Order**: Ensure the new handler doesn't interfere with ExtractionError handling
   - Keep ExtractionError handler first (more specific)
   - Add UnicodeDecodeError handler second
   - Both should be within the same try block

5. **Maintain Consistent Error Format**: Use the same emoji and formatting style as existing error messages
   - Start with "❌" emoji for consistency
   - Use bullet points for verification steps
   - Keep tone helpful and non-technical

## Testing Strategy

### Validation Approach

The testing strategy follows a two-phase approach: first, surface counterexamples that demonstrate the bug on unfixed code (UnicodeDecodeError crashes), then verify the fix catches the exception and returns user-friendly errors while preserving all successful extraction behavior.

### Exploratory Fault Condition Checking

**Goal**: Surface counterexamples that demonstrate the bug BEFORE implementing the fix. Confirm that invalid UTF-8 bytes cause crashes on unfixed code.

**Test Plan**: Create test cases that simulate extraction scenarios producing invalid UTF-8 bytes. Mock the decryption service to return non-UTF-8 bytes. Run these tests on the UNFIXED code to observe UnicodeDecodeError crashes and confirm the root cause.

**Test Cases**:
1. **Wrong Password Test**: Mock decryption to return `b'\x9b\xf2\xa3\xc4'` (invalid UTF-8) → will crash with UnicodeDecodeError on unfixed code
2. **Corrupted Data Test**: Mock decryption to return `b'\xff\xfe\xfd\xfc'` (invalid UTF-8) → will crash on unfixed code
3. **Partial Invalid UTF-8**: Mock decryption to return `b'Hello\x9b\xf2'` (starts valid, ends invalid) → will crash on unfixed code
4. **Empty Bytes Edge Case**: Mock decryption to return `b''` (empty bytes) → should decode successfully (no crash expected)

**Expected Counterexamples**:
- UnicodeDecodeError exceptions propagate uncaught, causing application crashes
- Error messages like: `UnicodeDecodeError: 'utf-8' codec can't decode byte 0x9b in position 0: invalid start byte`
- Possible causes: missing exception handler for UnicodeDecodeError, assumption that decryption always produces valid UTF-8

### Fix Checking

**Goal**: Verify that for all inputs where the bug condition holds (invalid UTF-8 from decryption), the fixed function catches the exception and returns a user-friendly error message.

**Pseudocode:**
```
FOR ALL input WHERE isBugCondition(input) DO
  result := extract_message_ui_fixed(input.stego_file, input.password)
  ASSERT result.startsWith("❌")
  ASSERT result.contains("Extraction failed") OR result.contains("decode")
  ASSERT result.contains("password") OR result.contains("modified") OR result.contains("embedded data")
  ASSERT NOT isinstance(result, Exception)
END FOR
```

### Preservation Checking

**Goal**: Verify that for all inputs where the bug condition does NOT hold (valid UTF-8 from decryption, or other error types), the fixed function produces the same result as the original function.

**Pseudocode:**
```
FOR ALL input WHERE NOT isBugCondition(input) DO
  ASSERT extract_message_ui_original(input) = extract_message_ui_fixed(input)
END FOR
```

**Testing Approach**: Property-based testing is recommended for preservation checking because:
- It generates many test cases automatically across the input domain (valid messages, various passwords, different image types)
- It catches edge cases that manual unit tests might miss (empty messages, special characters, long messages)
- It provides strong guarantees that behavior is unchanged for all non-buggy inputs

**Test Plan**: Observe behavior on UNFIXED code first for successful extractions and other error types, then write property-based tests capturing that behavior.

**Test Cases**:
1. **Successful Extraction Preservation**: Mock valid UTF-8 message extraction → observe successful decode on unfixed code → verify same behavior after fix
2. **ExtractionError Preservation**: Mock extraction_engine.extract() to raise ExtractionError → observe error handling on unfixed code → verify same error message after fix
3. **Password Validation Preservation**: Provide invalid password format → observe validation error on unfixed code → verify same error message after fix
4. **Image Validation Preservation**: Provide invalid image file → observe file reading error on unfixed code → verify same error message after fix

### Unit Tests

- Test UnicodeDecodeError is caught when decryption returns invalid UTF-8 bytes
- Test error message format matches expected user-friendly format
- Test error message contains actionable guidance (password, image integrity, embedded data)
- Test successful extraction continues to work with valid UTF-8 messages
- Test ExtractionError handling remains unchanged
- Test edge cases: empty bytes (valid UTF-8), single invalid byte, mixed valid/invalid bytes

### Property-Based Tests

- Generate random invalid byte sequences → verify all are caught and return error messages (not crashes)
- Generate random valid UTF-8 messages → verify all decode successfully with same format as original
- Generate random passwords and valid stego images → verify successful extraction behavior is preserved
- Test across many scenarios to ensure no regression in existing functionality

### Integration Tests

- Test full extraction flow with wrong password → verify user-friendly error instead of crash
- Test full extraction flow with modified stego image → verify graceful error handling
- Test full extraction flow with valid password and stego image → verify successful extraction unchanged
- Test that logging behavior is consistent (warnings for decode errors, info for successes)
