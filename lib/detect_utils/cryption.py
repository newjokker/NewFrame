import sys
import os
from Crypto.Cipher import AES
import struct


salt = 'txkj2019'


def aes_decrypt_file(key, in_filename, out_filename=None, chunksize=24 * 1024):
    """ Decrypts a file using AES (CBC mode) with the
        given key. Parameters are similar to encrypt_file,
        with one difference: out_filename, if not supplied
        will be in_filename without its last extension
        (i.e. if in_filename is 'aaa.zip.enc' then
        out_filename will be 'aaa.zip')
    """
    key = "{: <32}".format(key).encode("utf-8")

    if not out_filename:
        out_filename = os.path.splitext(in_filename)[0]

    with open(in_filename, 'rb') as infile:
        origsize = struct.unpack('<Q', infile.read(struct.calcsize('Q')))[0]
        iv = infile.read(16)
        decryptor = AES.new(key, AES.MODE_CBC, iv)

        with open(out_filename, 'wb') as outfile:
            while True:
                chunk = infile.read(chunksize)
                if len(chunk) == 0:
                    break
                outfile.write(decryptor.decrypt(chunk))

            outfile.truncate(origsize)

def decrypt_file(key, in_filename, out_filename=None, chunksize=24 * 1024):
    model_crypt_type = os.environ.get('MODEL_DATA_CRYPTION')
    if model_crypt_type == "WIBU_USB_SOFTWARE" or model_crypt_type == "WIBU_USB_HARDWARE":
        sys.path.append("/root/BGL-Release/BGL-platform/lib")
        import codemeter_cryption
        codemeter_cryption.wibu_cryp_file(in_filename, out_filename, "DEC")
    else:
        aes_decrypt_file(key, in_filename, out_filename, chunksize)

__all__ = [salt, decrypt_file]
