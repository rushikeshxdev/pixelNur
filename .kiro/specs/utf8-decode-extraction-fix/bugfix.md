# Bugfix Requirements Document

## Introduction

The Gradio app crashes with a `UnicodeDecodeError` when attempting to extract messages from stego images. The error occurs at line 265 in `pixelnur/app.py` when calling `message_bytes.decode('utf-8')` on the decrypted message bytes. This happens when the extraction or decryption process produces bytes that are not valid UTF-8, which can occur due to incorrect passwords, corrupted stego images, or extraction failures. The bug prevents users from receiving meaningful error messages and causes the application to crash instead of gracefully handling the error.

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN extraction returns corrupted bytes (due to wrong password, modified image, or extraction failure) THEN the system crashes with `UnicodeDecodeError: 'utf-8' codec can't decode byte 0x9b in position 0: invalid start byte`

1.2 WHEN decryption produces invalid UTF-8 bytes THEN the system raises an unhandled exception instead of providing a user-friendly error message

1.3 WHEN the UTF-8 decode fails THEN the user sees a raw Python exception instead of guidance on what went wrong

### Expected Behavior (Correct)

2.1 WHEN extraction returns corrupted bytes (due to wrong password, modified image, or extraction failure) THEN the system SHALL catch the `UnicodeDecodeError` and return a user-friendly error message explaining possible causes

2.2 WHEN decryption produces invalid UTF-8 bytes THEN the system SHALL detect the invalid encoding and provide actionable feedback to the user

2.3 WHEN the UTF-8 decode fails THEN the system SHALL display an error message that guides the user to verify their password, check image integrity, and confirm the image contains embedded data

### Unchanged Behavior (Regression Prevention)

3.1 WHEN extraction and decryption succeed with valid UTF-8 message THEN the system SHALL CONTINUE TO decode and display the message correctly

3.2 WHEN extraction fails with `ExtractionError` THEN the system SHALL CONTINUE TO handle it with the existing error message

3.3 WHEN password validation fails THEN the system SHALL CONTINUE TO return the password validation error before attempting extraction

3.4 WHEN image validation fails THEN the system SHALL CONTINUE TO return the image validation error before attempting extraction
