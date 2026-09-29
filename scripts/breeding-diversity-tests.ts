import { Population, type Member } from '../src/v2/population';
import { bodyFamily, closeRelatives, diverseFamily } from '../src/v2/breedingDiversity';
import { cloneGenome } from '../src/v2/genome';
import { mulberry32 } from '../src/v2/ops';
import { noveltyAcceptFloor, type ExploreMode } from '../src/v2/novelty';
import { Evolution } from '../src/v2/evolve';
import type { Store } from '../src/v2/store';
import type { Screener } from '../src/v2/screen';
import type { Phenotype } from '../src/v2/phenotype';

type Check = (name: string, ok: boolean, detail: string) => void;
export async function breedingDiversityTests(check: Check): Promise<void> {
  const seeds = Population.seeded(1);
  const base = seeds.get('G0-E13')!;
  const branch = seeds.get('G0-X46')!, fabric = seeds.get('G0-X48')!, linkage = seeds.get('G0-X50')!;
  const member = (template: Member, id: string): Member => ({ ...template, id, parents: [], energy: base.energy, species: base.species });
  // Deliberately put every member in one legacy species: the test must measure body
  // family selection, not the existing cross-species mating shortcut.
  const crowd = Array.from({length: 2000}, (_, i) => member(base, `G1-${String(i).padStart(4,'0')}`));
  const rare = [member(branch,'G1-2001'), member(fabric,'G1-2002'), member(linkage,'G1-2003')];
  const pool = [...crowd, ...rare];
  const counts = new Map<string, number>();
  const rng = mulberry32(1927);
  for (let i=0;i<1200;i++) {
    const picked = diverseFamily(pool, crowd[0], rng)!;
    const key = bodyFamily(picked[0].genome);
    counts.set(key,(counts.get(key)??0)+1);
  }
  check('diversity.rare-families', counts.size===3 && [...counts.values()].every(n=>n>300&&n<500), `2,000 familiar presets and 3 rare parents: family picks ${JSON.stringify([...counts])}`);

  const pop = new Population();
  pool.forEach(m=>pop.members.set(m.id,m));
  const before = JSON.stringify(pop.toJSON());
  const rates: number[] = [];
  for (const mode of ['off','gentle','explore','wild'] as ExploreMode[]) {
    const r=mulberry32(573); let different=0;
    for(let i=0;i<200;i++) {
      const [a,b]=pop.pickParents(base.energy,r,3,undefined,mode)!;
      if(bodyFamily(a.genome)!==bodyFamily(b.genome)) different++;
    }
    rates.push(different/200);
  }
  check('diversity.mode-strength', rates[0]<0.1 && rates[1]>0.15 && rates[1]<0.4 && rates[2]>0.45 && rates[3]>0.7 && rates.every((v,i)=>i===0||v>rates[i-1]), `cross-family pair rates Off / Gentle / Explore / Wild: ${rates.join(' / ')}`);
  check('diversity.non-destructive', before===JSON.stringify(pop.toJSON()), 'selection preserves every stored genome, vote, name and lineage');

  const sibling = {...rare[0],parents:['shared']};
  const anchor = {...crowd[0],parents:['shared']};
  const disliked = {...rare[1],dislikes:10};
  const eligible = diverseFamily([anchor,sibling,disliked,rare[2]],anchor,()=>0)!;
  check('diversity.lineage-and-votes', closeRelatives(anchor,sibling) && eligible.length===1 && eligible[0]===rare[2], 'prefers unrelated families and excludes clearly disliked rare candidates');
  const tiny = new Population(); tiny.members.set(anchor.id,anchor); tiny.members.set(crowd[1].id,crowd[1]);
  const fallback=tiny.pickParents(base.energy,mulberry32(10),3,undefined,'wild');
  check('diversity.small-population', !!fallback && fallback[0].id!==fallback[1].id && diverseFamily([anchor,crowd[1]],anchor,()=>0)===null, 'a single-family population still gets two distinct parents');
  rare.forEach(m=>m.hidden=true);
  const noHidden=Array.from({length:30},()=>pop.pickParents(base.energy,rng,3,undefined,'wild')!).every(pair=>pair.every(m=>!m.hidden));
  check('diversity.hidden',noHidden,'hidden rare parents are never selected');

  const tinted=cloneGenome(branch.genome); tinted.palette.p.hue=0.123;
  const composite=cloneGenome(branch.genome); composite.bodies.push(fabric.genome.bodies[0]);
  const reverse=cloneGenome(composite); reverse.bodies.reverse();
  check('diversity.family-identity',bodyFamily(tinted)===bodyFamily(branch.genome) && bodyFamily(composite)===bodyFamily(reverse) && bodyFamily(composite)!==bodyFamily(branch.genome),'colour variants share a family; layer order does not invent extra families');
  check('diversity.large-archive-floor',noveltyAcceptFloor('wild',2000)>0.18 && noveltyAcceptFloor('wild',1e9)>=0.18 && noveltyAcceptFloor('off',2000)===0,'large populations retain a meaningful novelty requirement');

  // Exercise the real breeding retry loop: after three rejections it must not
  // silently accept the fourth child. GPU measurements are stubbed, not selection.
  const evolution = new Evolution({} as Store, {
    screen: async()=>({ok:true,descriptor:[],metrics:{reactivity:0.8,hitLift:0.4,events:0.4}}),
  } as unknown as Screener);
  evolution.pop=new Population(); evolution.rng=mulberry32(813); evolution.changed=()=>{};
  let checks=0, adopted=0;
  const pheno={
    fingerprint:async()=>[1], duplicateOf:()=>null,
    acceptNovelty:()=>{checks++;return {ok:false,rel:0.01,floor:0.2};},
    adopt:()=>{adopted++;},mode:'wild',
  };
  evolution.pheno=pheno as unknown as Phenotype;
  const children=await evolution.breed([base,branch],1,'cross');
  check('diversity.no-retry-bypass',children.length===0 && checks>3 && adopted===0 && evolution.breeding===0,`${checks} novelty checks; no familiar children accepted on late retries`);
  checks=0; evolution.rng=mulberry32(813);
  pheno.acceptNovelty=()=>{checks++;return {ok:checks>=5,rel:checks>=5?0.8:0.01,floor:0.2};};
  const novel=await evolution.breed([base,branch],1,'cross');
  check('diversity.late-novel-child',novel.length===1 && checks===5 && adopted===1,'a genuinely novel child on the fifth attempt is still accepted');
}
