"""Verify the pre-registration hashes quoted in the reports.
PREREG1/2: the hash is of the PREREG string embedded in the notebook code (= file content without the final 'sha256:' line).
PREREG3/4/5: the hash is of the whole file. PREREG0 (pilot) was not hashed at the time; its timestamp is inside the file."""
import hashlib, pathlib, re, sys
here = pathlib.Path(__file__).parent
expected = {"PREREG1.txt": "1034f794335657cd5ba89c985694f59ad013f54a2acd96377a822323cc807df4",
            "PREREG2.txt": "9177a4a9795761ef3d358f8606789725e9169e543165ce91fb2676fa11e95589",
            "PREREG3.txt": "8292aeae6d00bb321ce34b40a9e4991969f4fddeb21f11ae4dcf64e0884e7bea",
            "PREREG4.txt": "95c74926234243bf45263b8a7e568795a9b411a5817aaac44715c08293e55282",
            "PREREG5.txt": "8f9732403eaf4427f98fd3370a39ed56d4fba571f9e306b265c4ed8b9737e121"}
ok = True
for f, h in expected.items():
    raw = (here / f).read_text(encoding="utf-8")
    if f in ("PREREG1.txt", "PREREG2.txt"):
        body = re.sub(r"\nsha256: [0-9a-f]{64}\n$", "", raw)
        got = hashlib.sha256(body.encode()).hexdigest()
    else:
        got = hashlib.sha256(raw.encode()).hexdigest()
    print(f"{f}: {'OK ' if got == h else 'MISMATCH'} {got}"); ok &= got == h
sys.exit(0 if ok else 1)
