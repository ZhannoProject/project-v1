#ifndef UNICODE
#define UNICODE
#endif
#ifndef _UNICODE
#define _UNICODE
#endif
#include <windows.h>
#include <wchar.h>
#include <stdio.h>

/* A relocatable launcher, compiled for ARM64 or x64 by gui/make_windows.py. Python remains a real interpreter, so the
 * GUI's training/evaluation subprocesses continue to work normally. */
int WINAPI wWinMain(HINSTANCE instance, HINSTANCE previous, PWSTR args, int show) {
    wchar_t root[32768], python[32768], project[32768], command[65536];
    DWORD length = GetModuleFileNameW(NULL, root, 32768);
    if (!length || length >= 32768 || !wcsrchr(root, L'\\')) return 1;
    *wcsrchr(root, L'\\') = 0;
    if (wcslen(root) > 30000) return 1;
    swprintf(python, 32768, L"%ls\\runtime\\pythonw.exe", root);
    swprintf(project, 32768, L"%ls\\app", root);
    swprintf(command, 65536, L"\"%ls\" -X utf8 -u -m gui.windows_start %ls", python, args);
    STARTUPINFOW startup = {0};
    PROCESS_INFORMATION process = {0};
    startup.cb = sizeof(startup);
    if (!CreateProcessW(python, command, NULL, NULL, FALSE, CREATE_NO_WINDOW,
                        NULL, project, &startup, &process)) {
        MessageBoxW(NULL, L"Unable to start EEG Training GUI. Extract the whole ZIP first.\n"
                         L"Keep the EXE, runtime and app folders together in a writable folder.",
                    L"EEG Training GUI", MB_OK | MB_ICONERROR);
        return 1;
    }
    CloseHandle(process.hThread);
    WaitForSingleObject(process.hProcess, INFINITE);
    DWORD result = 1;
    GetExitCodeProcess(process.hProcess, &result);
    CloseHandle(process.hProcess);
    return (int)result;
}
