def get_stats(hook, mod, inp, outp):
    hook.mean, hook.std = outp.mean().item(), outp.std().item()

def activation_stats(hook, mod, inp, outp):
    if not hasattr(hook, 'stats'):
        hook.stats = ([], [], [])
    means, stdevs, hists = hook.stats
    means.append(outp.data.mean().cpu())
    stdevs.append(outp.data.std().cpu())
    hists.append(outp.data.cpu().histc(40, 0, 10))
