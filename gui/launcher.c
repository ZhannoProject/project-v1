/*
 * Native launcher for "EEG Training GUI.app" (compiled by gui/make_app.py).
 *
 * Why a Mach-O and not a shell script: macOS attributes folder-access
 * permissions (Desktop, Documents, ...) to the *responsible* process — the
 * app LaunchServices started.  A script that `exec`s Python hands its
 * identity to Python.app, so the permission prompt is never shown as ours
 * and the venv under ~/Desktop cannot be read.  This binary stays alive
 * with the bundle's own signature and runs Python as a child, so the
 * prompt reads "EEG Training GUI would like to access files in your
 * Desktop folder" and the decision sticks to the app.
 *
 * Reads Contents/Resources/launch.cfg:
 *     root=/path/to/project v1
 *     python=/path/to/env/bin/python
 * then chdir(root) and runs `python -m gui --native`, appending its output
 * to ~/Library/Logs/eeg-training-gui.log.  On failure an alert points at
 * that log.
 */
#include <errno.h>
#include <fcntl.h>
#include <libgen.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

static int read_cfg(const char *path, char *root, char *python, size_t n) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char line[PATH_MAX + 16];
    root[0] = python[0] = '\0';
    while (fgets(line, sizeof line, f)) {
        line[strcspn(line, "\r\n")] = '\0';
        if (!strncmp(line, "root=", 5)) strlcpy(root, line + 5, n);
        else if (!strncmp(line, "python=", 7)) strlcpy(python, line + 7, n);
    }
    fclose(f);
    return (root[0] && python[0]) ? 0 : -1;
}

static void alert(const char *title, const char *msg) {
    char script[2048];
    snprintf(script, sizeof script,
             "display alert \"%s\" message \"%s\" as critical buttons {\"OK\"} default button \"OK\"",
             title, msg);
    char *argv[] = {"/usr/bin/osascript", "-e", script, NULL};
    pid_t pid;
    if (posix_spawn(&pid, argv[0], NULL, NULL, argv, environ) == 0) waitpid(pid, NULL, 0);
}

int main(void) {
    char exe[PATH_MAX], real[PATH_MAX], macos[PATH_MAX], contents[PATH_MAX];
    uint32_t size = sizeof exe;
    if (_NSGetExecutablePath(exe, &size) != 0 || !realpath(exe, real)) return 1;
    strlcpy(macos, dirname(real), sizeof macos);          /* .../Contents/MacOS */
    strlcpy(contents, dirname(macos), sizeof contents);   /* .../Contents       */

    char cfg[PATH_MAX], root[PATH_MAX], python[PATH_MAX], log[PATH_MAX];
    snprintf(cfg, sizeof cfg, "%s/Resources/launch.cfg", contents);
    if (read_cfg(cfg, root, python, sizeof root) != 0) {
        alert("EEG Training GUI could not start", "Contents/Resources/launch.cfg is missing or incomplete. Rebuild with: python -m gui.make_app");
        return 1;
    }
    const char *home = getenv("HOME");
    snprintf(log, sizeof log, "%s/Library/Logs/eeg-training-gui.log", home ? home : "/tmp");

    if (chdir(root) != 0) {
        char msg[PATH_MAX + 200];
        snprintf(msg, sizeof msg, "Cannot open the project folder:\n%s\n(%s)\n\nIf macOS asked for folder access, allow it in System Settings > Privacy & Security > Files and Folders. Rebuild with python -m gui.make_app if the project moved.", root, strerror(errno));
        alert("EEG Training GUI could not start", msg);
        return 1;
    }
    setenv("PYTHONUNBUFFERED", "1", 1);

    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_addopen(&fa, 1, log, O_WRONLY | O_CREAT | O_APPEND, 0644);
    posix_spawn_file_actions_adddup2(&fa, 1, 2);
    char *argv[] = {python, "-m", "gui", "--native", NULL};
    pid_t pid;
    int rc = posix_spawn(&pid, python, &fa, NULL, argv, environ);
    posix_spawn_file_actions_destroy(&fa);
    if (rc != 0) {
        char msg[PATH_MAX + 200];
        snprintf(msg, sizeof msg, "Cannot run the project's Python:\n%s\n(%s)\n\nRebuild with python -m gui.make_app if the venv moved.", python, strerror(rc));
        alert("EEG Training GUI could not start", msg);
        return 1;
    }
    int status = 0;
    waitpid(pid, &status, 0);
    int code = WIFEXITED(status) ? WEXITSTATUS(status) : 1;
    if (code != 0) {
        char msg[PATH_MAX + 300];
        snprintf(msg, sizeof msg, "Python exited with code %d. Details are in\n%s\n\nMost common cause: macOS denied access to the project folder. Allow it in System Settings > Privacy & Security > Files and Folders (or Full Disk Access) for EEG Training GUI, then open the app again.", code, log);
        alert("EEG Training GUI stopped", msg);
    }
    return code;
}
