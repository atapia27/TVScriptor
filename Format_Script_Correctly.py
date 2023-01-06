"""
Takes a full script, formatting it during the initial process. Ensures that
formatting does not affect Machine Learning.

Finds the longest word in given text, which helps determine the character size (maxlen)
property for our semi-redundant sequences
"""

def correct_format(string):
    # split the string into lines
    lines = string.split('\n')

    # initialize an empty result string
    result = ''

    # iterate through the lines
    for line in lines:
        # if the line starts with a name followed by a ':'
        if first_word_ends_with_colon(line):
            # add the line to the result string
            result += line
            # add an empty line before the line
            result += '\n'

    return result


def first_word_ends_with_colon(line):
    # split the line into words
    words = line.split()

    # if the line is not empty and has at least one word
    if words and words[0]:
        # check if the first word ends with a ':' character
        return words[0][-1] == ':'
    # if the line is empty or has no words, return False
    else:
        return False


transcript = """Michael: All right Jim. Your quarterlies look very good. How are things at the library?
Jim: Oh, I told you. I couldn't close it. So...

Michael: So you've come to the master for guidance? Is this what you're saying, grasshopper?

Jim: Actually, you called me in here, but yeah.

Michael: All right. Well, let me show you how it's done.
Michael: [on the phone] Yes, I'd like to speak to your office manager, please. Yes, hello. This is Michael Scott. I am the Regional Manager of Dunder Mifflin Paper Products. Just wanted to talk to you manager-a-manger. [quick cut scene] All right. Done deal. Thank you very much, sir. You're a gentleman and a scholar. Oh, I'm sorry. OK. I'm sorry. My mistake. [hangs up] That was a woman I was talking to, so... She had a very low voice. Probably a smoker, so... [Clears throat] So that's the way it's done.
Michael: I've, uh, I've been at Dunder Mifflin for 12 years, the last four as Regional Manager. If you want to come through here... See we have the entire floor. So this is my kingdom, as far as the eye can see. This is our receptionist, Pam. Pam! Pam-Pam! Pam Beesly. Pam has been with us for... forever. Right, Pam?
Pam: Well. I don't know.

Michael: If you think she's cute now, you should have seen her a couple of years ago. [growls]

Pam: What?

Michael: Any messages?

Pam: Uh, yeah. Just a fax.

Michael: Oh! Pam, this is from Corporate. How many times have I told you? There's a special filing cabinet for things from corporate.

Pam: You haven't told me.

Michael: It's called the wastepaper basket! Look at that! Look at that face.
Michael: People say I am the best boss. They go, "God we've never worked in a place like this before. You're hilarious." "And you get the best out of us." [shows the camera his WORLD'S BEST BOSS mug] I think that pretty much sums it up. I found it at Spencer Gifts.
Dwight: [singing] Shall I play for you? Pa rum pump um pum [Imitates heavy drumming] I have no gifts for you. Pa rum pump um pum [Imitates heavy drumming]
Jim: My job is to speak to clients on the phone about... uh, quantities and type of copier paper. You know, whether we can supply it to them. Whether they can pay for it. And... I'm boring myself just talking about this.
Michael: Whassup!
Jim: Whassup! I still love that after seven years.

Michael: Whassup!

Dwight: Whassup!

Michael: Whass...up!

Dwight: Whassup.

Michael: [Strains, grunts] What?

Jim: Nothing.

Michael: OK. All right. See you later.

Jim: All right. Take care.

Michael: Back to work.
Jan: [on her cell phone] Just before lunch. That would be great.
Michael: Corporate really doesn't really interfere with me at all. Jan Levinson Gould. [walking out of his office] Jan, hello. I call her Hillary Rodham Clinton. Right? Not to her face, because... well, not because I'm scared of her. Because I'm not. But, um... Yeah.
Jan: Alright, was there anything you wanted to add to the agenda?
Michael: Um... Me no get an agenda.

Jan: What? I'm sorry?

Michael: I didn't get any agenda.

Jan: Well, I faxed one over to you this morning.

Michael: Really? I didn't... [looks at Pam] Did we get a fax this morning?

Pam: Uh, yeah, the one...

Michael: Why isn't it in my hand? A company runs on efficiency of communication, right? So what's the problem, Pam? Why didn't I get it?

Pam: You put in the garbage can that was a special filing cabinet.

Michael: Yeah, that was a joke. That was a joke that was actually my brother's, and... It was supposed to be with bills and it doesn't work great with faxes.

Jan: Do you want to look at mine?

Michael: Yeah, yeah. Lovely. Thank you.

Jan: OK. Since the last meeting, Ellen and the board have decided we can't justify a Scranton branch and a Stamford branch.

Michael: OK...

Jan: Michael, don't panic.

Michael: No, no, no, no, this is good. This is good. This is fine. Excellent.

Jan: No, no, no, Michael, listen OK. Don't panic. We haven't made... We haven't decided.

Michael: All the alarm bells are kind of going... ringie-dingie-ding!

Jan: I've spoken to Josh in Stamford. I've told him the same as you and it's up to either him or you to convince me that your branch can incorporate the other.

Michael: OK. No problem.

Jan: This does, however, mean that there is going to be downsizing.

Michael: Me no wanna hear that, Jan. Because downsizing is a bitch. It is a real bitch. And I wouldn't wish that on Josh's men. I certainly wouldn't wish it on my men. Or women, present company excluded. Sorry. Uh, is Josh concerned about downsizing himself? Not downsizing himself but is he concerned about downsizing?
Michael: Question. How long do we have to... [Telephone rings] Oh uh, Todd Packer, terrific rep. Do you mind if I take it?
Jan: Go ahead.

Michel: Packman.

Todd Packer: Hey, you big queen.

Michael: Oh, that's not appropriate.

Todd Packer: Hey, is old Godzillary coming in today?

Michael: Uh, I don't know what you mean.

Todd Packer: I've been meaning to ask her one question. Does the carpet match the drapes?

Michael: Oh, my God! Oh! That's... horrifying. Horrible. Horrible person.

Jan: So do you think we could keep a lid on this for now? I don't want to worry people unnecessarily.

Michael: No, absolutely. Under this regime, it will not leave this office. [zips his lips] Like that.
Phyllis: So what does downsizing actually mean?
Stanley: Well...
Oscar: You guys better update your resumes just like I'm doing.
Angela: I bet it's gonna be me. Probably gonna be me.
Kevin: Yeah, it'll be you.
Pam: I have an important question for you.
Jim: Yes?

Pam: Are you going to Angela's cat party on Sunday?

Jim: Yeah, stop. That is ridiculous.
Michael: Am I going to tell them? No, I am not going to tell them. I don't see the point of that. As a doctor, you would not tell a patient if they had cancer.
Michael: Hey.
Ryan: Hey.

Pam: This is Mr. Scott.

Michael: Guilty! Guilty as charged!

Ryan: Ryan Howard from the temp agency. Daniqua sent me down to start today.

Michael: Howard, like Moe Howard. Three Stooges.

Ryan: Yup.

Michael: Watch this. This is Moe. Nyuck-nyuck-nyuck-nyuck-nyuck. Mee! [hold hand up for a high five] Ah, right here. Three Stooges. Oh, Pam. It's a guy thing, Pam. I'm sort of a student of comedy. Watch this. Here we go. [Yelling in cod German] I'm h*tler. Adolf h*tler. [Continues with cod German]
Pam: I don't think it would be the worst thing if they let me go because then I might... I don't think it's many little girls' dream to be a receptionist. I like to do illustrations. Um... Mostly watercolor. A few oil pencil. Um, Jim thinks they're good.
Pam: Dunder Mifflin. This is Pam.
Jim: Sure. Mr. Davis, let me call you right back. Yeah, something just came up. Two minutes. Thank you very much. Dwight, what are you doing?
Dwight: What?

Jim: What are you doing?

Dwight: Just clearing my desk. I can't concentrate.

Jim: It's not on your desk.

Dwight: It's overlapping. It's all spilling over the edge. One word, two syllables. Demarcation.

Dwight: You can't do that.

Jim: Why not?

Dwight: Safety violation. I could fall and pierce an organ.

Jim: [crosses fingers] We'll see. [Dwight begins smashing pencils with his phone] This is why the whole downsizing thing just doesn't bother me.

Dwight: Downsizing?
Dwight: Downsizing? I have no problem with that. I have been recommending downsizing since I first got here. I even brought it up in my interview. I say, bring it on.
Pam: You just still have these messages from yesterday.
Michael: Relax. Everything's under control. Uh, yeah. Yeah. That's important. Right. Oh this is so important, I should run to answer it. [Imitating Six-Million Dollar Man sound effect]

Pam: What?

Michael: Come on. Six-Million Dollar Man! Steve Austin! Actually, that would be a good salary for me, don't you think? Six million dollars? Memo to Jan. I deserve a raise.

Pam: Don't we all?

Michael: I'm sorry?

Pam: Nothing.

Michael: If you're unhappy with your compensation, maybe you should take it up with HR. OK. Not today, OK? Pam, just be professional. [Sighs]
Michael: I think I'm a role model here. I think I garner people's respect. [Imitating a PA] Attention all Dunder Mifflin employees, please. We have a meeting in the conference room, ASAP.
Michael: People I respect, heroes of mine, would be Bob Hope... Abraham Lincoln, definitely. Bono. And probably God would be the fourth one. And I just think all those people really helped the world in so many ways that it's really beyond words. It's really incalculable.
Michael: Now I know there's some rumors out there and I just kind of want to set the record straight.
Dwight: I'm Assistant Regional Manager. I should know first.

Michael: Assistant to the Regional Manager.

Dwight: OK, um, can you just tell me please? Just tell me quietly. Can you whisper it in my ear?

Michael: I'm about to tell everybody. I'm just about to tell everybody.

Oscar: Can't you just tell us.

Dwight: Please, OK? Do you want me to tell 'em?

Michael: You don't know what it is. [Laughs]

Dwight: OK. You tell 'em. With my permission. Permission granted.

Michael: I don't need your permission.

Dwight: Go ahead.

Michael: Corporate has deemed it appropriate to enforce an ultimatum upon me. And Jan is thinking about downsizing either the Stamford branch or this branch.
Ryan: I heard they might be closing this branch down. That's just the rumor going around. This is my first day. I don't really know.
Oscar: Yeah but Michael, what if they downsize here?
Michael: Not gonna happen.

Stanley: It could be out of your hands Michael.

Michael: It won't be out of my hands Stanley, OK. I promise you that.

Stanley: Can you promise that?

Dwight: On his mother's grave.

Michael: No.

Phyllis: What?

Michael: Well, yeah, it is a promise. And frankly, I'm a little insulted that you have to keep asking about it.

Stanley: It's just that we need to know.

Michael: I know. Hold on a second. I think Pam wanted to say something. Pam, you had a look that you wanted to ask a question just then.

Pam: I was in the meeting with Jan and she did say that it could be this branch that gets the axe.

Man: Are you sure about that?

Michael: Well, Pam maybe you should stick to the ongoing confidentiality agreement of meetings.

Dwight: Pam, information is power.

Stanley: You can't say for sure whether it'll be us or them, can you?

Michael: No, Stanley. No, you did not see me in there with her. I said if Corporate wants to come in here and interfere, then they're gonna have to go through me. Right? You can go mess with Josh's people, but I'm the head of this family, and you ain't gonna be messing with my chillin.
Jim: If I left, what would I do with all this useless information in my head? You know? Tonnage price of manila folders? Um, Pam's favorite flavor of yogurt, which is mixed berry.
Pam: Jim said mixed berries? Well, yeah, he's on to me. Um... [Laughs]
Michael: Watch out for this guy. Dwight Schrute in the building. This is Ryan, the new temp.
Ryan: What's up? Nice to meet you.

Michael: Introduce yourself. Be polite.

Dwight: Dwight Schrute, Assistant Regional Manager.

Michael: Assistant to the Regional Manager. So, uh, Dwight tell him about the kung fu and the car and everything.

Dwight: Uh... yeah I got a '78 280Z. Bought it for $1,200. Fixed it up. It's now worth three grand.

Michael: That's his profit.

Dwight: New engine, new suspension, I got a respray. I've got some photos.
Dwight: Damn it! Jim!
Michael: OK. Hold on, hold on. The judge is in session. What is the problem here?

Dwight: He put my stuff in Jell-O again.

Pam: [Laughing]

Dwight: That's real professional thanks. That's the third time and it wasn't funny the first two times either Jim.
Dwight: It's OK here, but people sometimes take advantage because it's so relaxed. I'm a volunteer Sheriff's Deputy on the weekends. And you cannot screw around there. That's sort of one of the rules.
Michael: What is that?
Dwight: That is my stapler.

Michael: No, no, no. Do not take it out. You have to eat it out of there, because there are starving people in the world [turns to camera] which I hate, and it is a waste of that kind of food.

Dwight: OK you know what, you can be a witness. [points to Ryan] Can you reprimand him?

Jim: How do you know it was me?

Dwight: It's always you. Are you going to discipline him or not?

Michael: Discipline. Kinky! [Laughs] All right, here's the deal you guys. The thing about a practical joke is you have to know when to start and as well as when to stop.

Dwight: Yeah.

Michael: And yeah, Jim this is the time to stop putting Dwight's personal effects into Jell-O.

Jim: OK. Dwight, I'm sorry, because I have always been your biggest flan.

Michael: [Laughing] Nice. That's the way it is around here. It just kind of goes round and round.

Ryan: You should've put him in custardy.

Michael: Hey! Yes! New guy! He scores.

Dwight: OK, that's great. I guess what I'm most concerned with is damage to company property. That's all.

Michael: Pudding. Pudding... I'm trying to think of another dessert to do.
Jim: Do you like going out at the end of the week for a drink?
Pam: Yeah.

Jim: That's why we're all going out. So we can have an end-of-the-week-drink.

Pam: So when are we going out?

Jim: Tonight, hopefully.

Pam: OK. Yeah.

Roy: Hey, man.

Jim: What's going on?

Roy: Hey, baby.

Pam: Hey.
Pam: Roy's my fiance. We've been engaged about three years. We were supposed to get married in September but I think we're gonna get married in the spring.
Pam: Do you mind if I go out for a drink with these guys?
Roy: No, no. Come on. Let's get out of here and go home.

Pam: OK. I'm gonna be a few minutes. So it's only twenty past five. I still have to do my faxes.

Jim: You know what? You should come with us. Because you know we're all going out and it could be a good chance for you to see what people are like outside the office. I think it could be fun.

Roy: It sounds good. Seriously, we've gotta get going.

Jim: Yeah, yeah.

Jim: Um... What's in the bag?

Roy: Just tell her I'll talk to her later.

Jim: No, definitely. All right, dude. Awesome. Will do.
Jim: Do I think I'll be invited to the wedding? [scratches head]
Michael: So have you felt the vibe yet? We work hard, we play hard. Sometimes we play hard when we should be working hard. Right? I guess the atmosphere that I've created here is that I'm a friend first, and a boss second... and probably an entertainer third. [Knock at door] Just a second. Right? Oh, hey do you like The Jamie Kennedy Experiment? Punk'd and all that kind of stuff?
Ryan: Yeah.

Michael: You are gonna be my accomplice. Just go along with it, OK?

Ryan: All right.

Michael: Just follow my lead. Don't pimp me, all right? Come in. So, uh, Corporate just said that I don't want to...

Pam: You got a fax.

Michael: Oh, thank you. Pam, can you come in here for a sec. Just have a seat. I was gonna call you in anyway. You know Ryan. As you know, there is going to be downsizing. And you have made my life so much easier in that I am going to have to let you go first.

Pam: What? Why?

Michael: Why? Well, theft and stealing.

Pam: Stealing? What am I supposed to have stolen?

Michael: Post-it Notes.

Pam: Post-it Notes? What are those worth, 50 cents?

Michael: 50 cents, yeah. If you steal a thousand Post-It Notes at 50 cents apiece, and you know, you've made a profit... margin. You're gonna run us out of business, Pam.

Pam: Are you serious?

Michael: Yeah. I am.

Pam: I can't believe this. I mean I have never even stolen as much as a paperclip and you're firing me.

Michael: But the best thing about it is that we're not going to have to give you any severance pay. Because that is gross misconduct and... Just clean out your desk. I'm sorry.

Michael: [Pam starts crying] You been X'd punk! [Laughing] Surprise! It's a joke. We were joking around. See? OK. He was in on it. He was my accomplice. And it was kind of a morale booster thing. And we were showing the new guy around, giving him the feel of the place. So you... God, we totally got you.

Pam: You're a jerk.

Michael: I don't know about that.
Michael: What is the most important thing for a company? Is it the cash flow? Is it the inventory? Nuh-uh. It's the people. The people. My proudest moment here was not when I increased profits by 17% or when I cut expenses without losing a single employee. No, no, no, no, no. It was a young Guatemalan guy. First job in the country, barely spoke English. He came to me, and said, "Mr. Scott, would you be the godfather of my child?" Wow. Wow. Didn't work out in the end. We had to let him go. He sucked.
Pam: Hey.
Jim: Hey.

Jim: How are things?

Pam: Good. I thought you were going out for a drink with...

Jim: Oh no, I just decided not to. How's your headache?

Pam: It's better, thanks.

Jim: Good. Good.

Pam: Yeah.

Jim: That's great

Pam: Is...?

Jim: Yeah?

Pam: Um... Are you...

Jim: Am I walking out?

Pam: Yes.

Jim: Yes, I... Do you want to...

Pam: Yeah.

Jim: Great. Let me just...

Jim: [Car horn honking] Oh, Roy.

Pam: Yeah. Listen, have a nice weekend.

Jim: Yeah, definitely. You too. Enjoy it. [looks at camera] You know what, just come here.
Michael: Hey, uh, can I help you out in here?
Mr. Brown: Oh, I'm all set, thanks.

Michael: Gotcha. Good. I'd go with the rows. That's a good idea.
Michael: Today is diversity day and someone's going to come in and talk to us about diversity. It's something that I've been pushing, that I've been wanting to push, for a long time and Corporate mandated it. And I never actually talked to Corporate about it. They kind of b*at me to the punch, the bastards. But I was going to. And I think it's very important that we have this. I'm very, very excited.
Jim: That's the thing. It's very sturdy paper and on the back it says, "100% post-consumer content." What? Hello? Uh-huh. Wait. What? I'm sorry, Mr. Decker. I think I'm losing you. [Shedder whirring] Hello? Hello? Yeah. Hold on one second. I don't know. Hold on one second.
Jim: Do you really have to do that right now?

Dwight: Yes I do. I should have done it weeks ago actually.

Jim: Mr. Decker, I'm sorry about that. What were you... Can you hold on one second? Yeah, just one second. Thanks. [Power off, silence] Hello? That's it. Perfect. So what I was saying... [Dialing tone] Hello? Thanks, Dwight.

Dwight: Retaliation. Tit for tit.

Jim: That is not the expression.

Dwight: Well, it should be.
Jim: This is my biggest sale of the year. They love me over there for some reason. I'm not really sure why but I make one call over there every year, just to renew their account, and that one call ends up being 25% of my commission for the whole year, so I buy a mini bottle of champagne, celebrate a little. And this year I'm pushing recycled paper on them for one percent more. I know. I'm getting cocky. Right?
Jim: Solitaire?
Pam: Yeah, Freecell.

Jim: Six on seven.

Pam: I know. I saw that.

Jim: So then, why didn't you do it?

Pam: I'm saving that 'cause I like it when the cards go T-ts-ts-tch-tch-tch.

Jim: Who doesn't love that?
Michael: Hey, Oscar! How are you doing, man?
Oscar: All right.

Michael: Did you have a good weekend going there?

Oscar: It was fine.

Michael: Oh yeah, I bet it was fun. [to Mr. Brown] Oh, hey! This is Oscar---

Oscar: Martinez.

Michael: Right. See? I don't even know, first-name basis!

Mr. Brown: Great. We're all set.

Michael: Oh hey, well, diversity, everybody, let's do it. Oscar works in... here. Jim, could you wrap it up, please?

Jim: Yeah, uh, Mr. Decker, please.

Michael: It's diversity day, Jim. I wish every day was diversity day.

Jim: You know what? I'm actually going to have to call you back. Thank you. Sorry about that.
Mr. Brown: Thank you. Thank you. Thank you. Great.
Michael: Come on people, let's get 'em in. Get in the cards! Get in the cards!

Mr. Brown: Thank you. Thank you very much. OK. Thanks for filling these out and I promise this'll be quick. At Diversity Today, our philosophy is about honesty and positive expectations. We believe that 99% of the problems in the workplace arise simply out of ignorance.

Michael: You know what? This is a color-free-zone here. Stanley, I don't look at you as another race.

Mr. Brown: Uh, see this is what I'm talking about. We don't have to pretend we're color-blind.

Michael: Exactly, were not...

Mr. Brown: That's fighting ignorance with more ignorance.

Michael: With tolerance.

Mr. Brown: No. With more ignorance.

Michael: Ignorance.

Mr. Brown: Right. Exactly. Uh, instead, we need to celebrate our diversity.

Michael: Let's celebrate.

Mr. Brown: Right. OK.

Michael: Celebrate good times. Come on! Let's celebrate diversity. Right?

Mr. Brown: Yes, exactly. Now here's what we're going to do. I've noticed that...

Michael: You know what? Here's what we're going to do. Why don't we go around and everybody... everybody say a race that you are attracted to sexually. I will go last. Go.

Dwight: I have two. White and Indian.

Mr. Brown: Actually, I'd prefer not to start that way. Michael, I would love to have your permission to run this session. Can I have your permission?

Michael: Yes.

Mr. Brown: Thank you very much. And it would also help me if you were seated.

Michael: OK.

Mr. Brown: Thank you. OK. Now, at the start of the session, I had you all write down an incident that you found offensive in the workplace. Now, what I'm going to do is choose one and we're going to act it out.

Dwight: A few of the ground rules?

Michael: Hey, hey why don't you run it by me and I'll run it by him.

Dwight: OK, can we steer away from gay people?

Mr. Brown: Um...

Dwight: I'm sorry. It's an orientation. It's not a race. Plus a lot of other races are intolerant of gays, so...paradox.

Mr. Brown: Well, we only have an hour.

Dwight: I figured it would save time.

Michael: OK. Why don't we just defer to Mr...

Mr. Brown: Mr. Brown.

Michael: Ah. Oh, right! OK. First test. I will not call you that.

Mr. Brown: Well, it's my name. It's not a test. OK? Um, so looking through the cards, I've noticed that many of you wrote down the same incident, which is ironic, because it's the exact incident I was brought in here to respond to. Now, how many of you are familiar with the Chris Rock routine? Very good. OK.
Michael: How come Chris Rock can do a routine and everybody finds it hilarious and ground-breaking and then I go and do the exact same routine, same comedic timing, and people file a complaint to Corporate? Is it because I'm white and Chris is black?
Mr. Brown: So we're going to reenact this with a more positive outcome.
Michael: I will play the Chris Rock guy. I would like to see someone else pull this off.

Mr. Brown: Well, let's have someone who wasn't involved in the reenactment.

Michael: OK, I will play guy listening.

Mr. Brown: Great. Guy listening. Ok, anyone else remember?

Kevin: I remember.

Mr. Brown: Great. You're the Chris Rock guy and you're guy listening.

Michael: OK.
Michael: Kevin is a great guy. He's a great accountant. He is not much of an entertainer.
Kevin: Basically, there are two types of black people and black people are actually more r*cist because they hate the other type of black people. Every time the one type wants to have a good time, then the other type comes in and makes a real mess.
Michael: OK. I'm sorry. I'm sorry. He's ruin... He's butchering it. Could you just let me... [As Chris Rock] Every time... Every time black people want to have a good time, some ignant ass... [Bleep] I take care of my kid!

Mr. Brown: Wait a second.

Michael: [Bleep] They always want credit for something they supPOSED to do!

Mr. Brown: Stop it!

Michael: [As Chris Rock] What you want a cookie?
Mr. Brown: Now, this is a simple acronym. HERO. Uh, at Diversity Today, we believe it is very easy to be a HERO. All you need are honesty, empathy, respect and open-mindedness.
Dwight: Excuse me, I'm sorry, but that's not all it takes to be a hero.

Mr. Brown: Oh, great. Well, what is a hero to you?

Dwight: A hero k*ll people, people that wish him harm.

Mr. Brown: OK.

Dwight: A hero is part-human and part-supernatural. A hero is born out of a childhood trauma, or out of a disaster that must be avenged.

Mr. Brown: Ok, you're thinking of a superhero.

Dwight: We all have a hero in our heart.

Mr. Brown: Now, I need you to take these forms. This kind of expresses the joint experience we had today. And I need you to look 'em over and sign them as kind of a group pledge.

Michael: [Clears throat] I don't think I can sign this.

Mr. Brown: I can't leave until you do.

Michael: Well, OK, it says here that I learned something and I knew all this stuff already, so... I know, I could sign something that says that I taught something, or that I helped you teach something, so... Pam! Where is she? Pam, could we change something on this?

Mr. Brown: Michael, can I talk to you candidly?

Michael: Sure.
Mr. Brown: We both know that I'm here because of the comments you made.
Michael: Here's the thing. This office, I think this is very advanced in terms of... racial awareness and it's probably more advanced than you're used to. That's probably throwing you off a little bit.

Mr. Brown: Um, it's not throwing me. I need your signature.

Michael: OK, well I know. You told me that several times.

Mr. Brown: Yes, but you're not listening to me. Yours is the only signature I need.

Michael: OK.

Mr. Brown: Those are my instructions from the Corporate offices to put you through this seminar for the comments that you made. The reason I made copies for everyone was so you wouldn't be embarrassed.

Michael: Well, here I am thinking that you actually cared about diversity training. And you don't.

Mr. Brown: Don't worry about dating.

Michael: I won't.

Mr. Brown: OK. Thank you.

Michael: Yeah, yeah.
Michael: "I regret my actions. I regret offending my coworkers. I pledge to bring my best spirit of honesty, empathy, respect and open-mindedness..." Open-mindedness, is that even a word? "...into the workplace. In this way, I can truly be a hero. Signed, Daffy Duck." [Laughing] He's going to lose it when he reads that.
Jim: Yeah, hi. Is Mr. Decker around? Oh, well, could you just have him call me after lunch? Thank you.
Michael: "I pledge to always keep an open mind and an open heart." I do believe... in that part of the pledge I that just read. But a pledge? Come on. I mean who are we, the Girl Scouts? No. Look... the guy, "Mr. Brown," he got us halfway there. He got us talking. Well, no. I got us talking. He got us nothing. He insulted us and he abandoned us. You call that diversity training? I don't. Were there any connections between any of us? Did anyone look each other in the eye? Was there any emotion going on? No. Where was the heart? I didn't see any heart. Where was my Oprah moment? OK, get as much done as you can before lunch because, afterward, I'm going to have you all in tears.
Michael: All right? Everybody pretty? Come on. Here we go. It's time. Let's do some good.
Toby: Hey, we're not all going to sit in a circle Indian style are we? [Laughing]

Michael: Get out.

Toby: I'm sorry.

Michael: No, this is not a joke. OK? That was offensive and lame. So double offensive. This is an environment of welcoming and you should just get the hell out of here. OK, let's go. Let's do it. Come on. Let's have some fun, everybody. Here we go. Take a seat. Cop a squat. And um... thanks for coming in. Um... Diversity... is the cornerstone of progress as I've always said. But don't take my word for it. Let's take a look at the tape.
Michael: [on the tape] Hi. I'm Michael Scott. I'm in charge of Dunder Mifflin Paper Products here in Scranton, Pennsylvania but I'm also the founder of Diversity Tomorrow, because today is almost over. Abraham Lincoln once said that, "If you're a r*cist, I will attack you with the North." And those are the principles that I carry with me in the workplace.
Michael: OK. Questions? Comments? Anybody? Jim?
Jim: [/b]: Uh, is that it?

Michael: Yes. I only had an hour to put it together but I'm going to add on to it later on.

Kevin: It was kind of hard to hear.

Michael: Uh, yes. That probably had something to do with the camera work. Anybody else? Um...

Kelly: I have a customer meeting.

Michael: Yeah, well, if you leave we'll only have two left. Yes. Enjoy. Absolutely. Namaste. Ok, well since I am leading this, let's get down to business and why don't I just kind of introduce myself, OK? Um. I am Michael and I am part English, Irish, German and Scottish. Sort of a virtual United Nations. But what some of you might not know is that I am also part Native American Indian.

Oscar: What part Native American?

Michael: Two fifteenths.

Oscar: Two fifteenths, that fraction doesn't make any sense.

Michael: Well, you know what, it's kind of hard for me to talk about it. Their suffering. So who else? Let's get this popping. Come on. Who's going? Who's going? Let's go here. Oscar, right here. You're on.

Oscar: OK, Michael, um... Both my parents were born in Mexico.

Michael: Oh, yeah...

Oscar: And, uh, they moved to the United Sates a year before I was born. So I grew up in the United States.

Michael: Wow.

Oscar: My parents were Mexican.

Michael: Wow. That is... That is a great story. That's the American Dream right there, right?

Oscar: Thank... Yeah...

Michael: Um, let me ask you, is there a term besides Mexican that you prefer? Something less offensive?

Oscar: Mexican isn't offensive.

Michael: Well, it has certain connotations.

Oscar: Like what?

Michael: Like... I don't... I don't know.

Oscar: What connotations, Michael? You meant something.

Michael: No. Now, remember that honesty...

Oscar: I'm just curious.

Michael: ...empathy, respect... [Phone ringing] Jim! Jim!

Jim: Hello? Hello?
Michael: I have something here. I want you to take a card. Put it on your fore... Don't look at the card. I want you to take the card and put it on your forehead and... Take a card, take a card, any card. Um... And I want you to treat other people like the race that is on their forehead. OK? So everybody has a different race. Nobody knows what their race is, so... I want you to really go for it, cause this is real. You know, this isn't just an exercise. This is real life. And... I have a dream that you will really let the sparks fly. Get 'er done.
Michael: Why? Because Martin Luther King is a hero of mine. There's this great Chris Rock bit about how streets named after Martin Luther King tend to be more violent. I'm not going to do it but it's...
Michael: Oh this is a good one.
Pam: Um, hi. How are you?

Stanley: Fine. How are you?

Pam: Great.

Michael: Push it.

Stanley: I admire your culture's success in America.

Pam: Thank you.

Michael: Good. Bom bom bom-bom bom. Come on Olympics of Suffering right here. Slavery versus the Holocaust. Come on.

Stanley: Who am I supposed to be?

Michael: No, that was inadvertent. We didn't actually plan that.

Dwight: Lots of cultures eat rice, doesn't help me.

Dwight: Um... Shalom. I'd like to apply for a loan.

Pam: That's nice, Dwight.

Dwight: OK, do me. Something stereotypical so I can get it really quick.

Pam: OK, I like your food.

Dwight: Outback steakhouse. [Australian accent] I'm Australian, mate!

Michael: Pam, come on. "I like your food." Come on stir the pot. Stir the melting pot, Pam! Let's do it. Let's get ugly. Let's get real.

Pam: OK. If I have to do this, based on stereotypes that are totally untrue, that I do not agree with, you would maybe not be a very good driver.

Dwight: Oh, man, am I a woman?
Michael: You'll notice I didn't have anybody be an Arab. I thought that would be too expl*sive. No pun intended. But I just though. "Too soon for Arabs." Maybe next year. Um... You know, the ball's in their court.
Jim: What are you watching?
Ryan: Chappelle's Show.

Jim: Really?

Ryan: I downloaded it on her computer. I hope she doesn't mind. She just had a lot of extra space.

Jim: No way. I think she likes this stuff.

Ryan: Great. She's cute, huh?

Jim: Yeah, you know, she's engaged, but...

Ryan: Oh, no, the girl in the... sketch.

Jim: Oh, yeah. She's hot.
Kevin: Hey.
Angela: Hey.

Kevin: You wanna go to the beach?

Angela: Sure.

Kevin: You wanna get high?

Angela: No.

Kevin: I think you do, mon.

Angela: Stop...

Michael: OK. All right. No. It's good. You just need to push it. You need to go a little bit further. All right. OK.
Michael: [Voice raised, Indian accent] Kelly, how are you?
Kelly: I just had the longest meeting.

Michael: Oh! Welcome to my convenience store. Would you like some googi googi? I have some very delicious googi, googi, only 99 cents plus tax. Try my googi, googi. [Lowering voice] Try my googi, googi. [High-pitched voice] Try my googi, googi. Try my... [slap!]

Michael: [trying not to cry] All right! All right! Yes! That was great, she gets it! Now she knows what it's like to be a minority.
Jim: [on the phone] Mr. Decker, we didn't lose your sale today, did we? Excellent. OK. Let me just get your... what's that? No, we didn't close last time. I just need your... Oh. W-What code were you given? Oh, OK. That's actually another salesman here. I can redo it if you want to do that. Oh, he gave you a discount? No, I don't blame you.
Michael: I just hated it when that guy was in here. Mr. Brown, if that was his real name. I mean, he had never met any of us before, and here he was telling us how to do our thing. I just wanted... I just wanted to do it our way. You know? On our own. Man I should have gotten some food.
Kevin: [Itialian accent]Maybe some spagh-etti.

Michael: Okay, Kevin. You can take that off that thing, OK? That would really, really have shown him up, wouldn't it? If I'd brought in some burritos or some colored greens. Or some pad Thai. I love pad Thai.

Stanley: It's collard greens.

Michael: What?

Stanley: It's collard greens.

Michael: That doesn't really make sense. Because you don't call them collared people, that's offensive. Hmmm... OK, well, it's after five. So... Thank you very much. Buena vista Oscar. Thank you. Good job. Oh, my man. Thank you Brazil. Nice.
Jim: [Pam is asleep, resting her head on Jim's shoulder] Um... Hey.
Pam: [stirs] Mmmm.

Jim: Hey.

Pam: Oh.

Jim: We can go.

Pam: Sorry.

Jim: That's fine.
Jim: Uh... Not a bad day.
Michael: Pam. Pamela. Pam-elama-ding-dong. Making copies.
Pam: I'm not making any copies.

Michael: Let's go. Messages. Stat. Lots to do, lots to do. Information superhighway.

Pam: Nothing new.

Michael: Lay them on me. What?

Pam: There's nothing new.

Michael: That's not what you said earlier.

Pam: Oh, do you want me to repeat the messages that I gave you before for the... [nods toward camera]
Michael: The most sacred thing I do is care and provide for my workers, my family. I give them money. I give them food. Not directly, but through the money. I heal them. Today, I am in charge of picking a great new health care plan. Right? That's what this is all about. Does that make me their doctor? Um... Yes, in a way. Yeah, like a specialist.
Jan: So, which health plan have you decided on?
Michael: I am going to go with the best, Jan. I am going to go with the one with the acupuncture, therapeutic massage, you know, the works.

Jan: Wait, acupuncture? None of the plans have acupuncture. Have you looked at them closely Michael?

Michael: I think it was you who didn't look closely enough at the Gold Plan.

Jan: The Gold Plan? I'm not even on that plan.

Michael: Well, I'd recommend it. It's very good.

Jan: Michael.

Michael: You gotta crack these things open.

Jan: You know the whole reason that we're doing this, is to save money. So you just need to pick a provider and choose the cheapest plan.

Michael: Well, that is kind of a tough assignment. Um... It won't be popular decision around the old orifice.

Jan: It's your job. So...

Michael: Well, it's a su1c1de mission, you know.

Jan: Michael... maybe... I mean...

Michael: There, there...

Jan: Sometimes a manager, like yourself, has to deliver the bad news to the employees. I do it all the time.

Michael: [scoffs] When have you ever done that?

Jan: I'm doing it right now. To you.
Jim: Last night on Trading Spouses, there's... have you seen it?
Pam: No. I have a life.

Jim: Interesting, what's that like?

Pam: You should try it sometime.

Jim: Wow. But then who would watch my TV? That is...

Pam: [laughs]... your problem.

Michael: Jimbo! Ha haaaaa. Ah.
Michael: There's a decision that needs to be made, and I'm having an unbelievably a busy day. So I'm going to let you pick a health care plan for our office and then explain it to your co-workers.
Jim: Gosh.

Michael: Yeah!

Jim: That is a great offer. Thank you. I really think I should be concentrating on sales.

Michael: Really?

Jim: Yeah. I just don't think this is the kind of task, that I... am going to do. You know who would be great for this?
Jim: Any time Michael asks me to do anything, I just tell him that Dwight should do it.
Dwight: Yes. I can do it. I'm your man.
Jim: Right now, this is just a job. If I advance any higher in this company then this would be my career. And uh, well, if this were my career, I'd have to throw myself in front of a train.
Dwight: OK, first, let's go over some parameters. How many people can I fire?
Michael: Ah, none. You're picking a health care plan.

Dwight: OK, we'll table that for the time being. Two, I'll need an office. I think the conference room should be fine.

Michael: You can use the conference room as a temporary workspace.

Dwight: [to self] Yes, I have an office. [to camera] Bigger than his.

Michael: Nope, you cannot use it.

Dwight: OK, I take it back, it's a workspace.

Michael: Temporary workspace. You can use it.

Dwight: Thank you.
Michael: If Dwight fails, then that is strike two, and good for me for, ah, for giving him a second chance. And if he succeeds, then, you know, no one will be prouder than I am. I groomed him. I made him what he is today. Unless he fails, and we've talked about that already.
Dwight: What did I do? I did my job. I slashed benefits to the bone. I saved this company money. Was I too harsh? Maybe. I don't believe in coddling people.
Dwight: In the wild, there is no health care. In the wild, health care is, "Ow, I hurt my leg. I can't run. A lion eats me and I'm dead." Well, I'm not dead. I'm the lion. You're dead.
Stanley: There's no dental, there's no vision, there's a $1,200 deductible.
Pam: Dunder Mifflin, this is Pam.
Michael: [on phone] Pam, Michael Scott. How's tricks?

Pam: Where are you?

Michael: Oh, I am in my office. I am swamped. I have work up to my ears I'm busy, busy, busy. Can't step away. I just wanted to check in and see how everybody's doing. Everybody cool out there?

Pam: Actually, people are really unhappy. Um, Dwight sent around this memo and people are freaking out 'cause the...

Michael: Pam! Whoa, whoa, I'm sorry, I'm sorry, I, I, I, I gotta go. I'm getting a call.

Pam: No you're not.

Michael: I have to make a call after I finish... my work. You know what? Uh, just don't let anybody in my office under any conditions today. I'm just too busy. Too swamped, you know? I am unreachable. I am incommunicado, capisce?

Pam: OK.

Michael: Thank you, Oh, gah, here we go again. Gotta go, I have to take this.

Pam: Still no one calling.
Pam: Dwight, what...
Dwight: Uh, knock, please. Please knock. This is an office.

Jim: It says "workspace".

Dwight: Same thing.

Jim: If it's the same thing, then why did you write "workspace"?

Dwight: Just knock, Please? As a sign of respect for your superior.

Jim: You are not my superior.

Dwight: Oh gee, then why do I have an office?

Jim: I thought it was a workspace?

Pam: OK. Dwight. Are you really in charge of picking the health care plan?

Dwight: Yes. And my decision in final.

Pam: This is a ridiculously awful plan. Because you cut everything.

Dwight: Aww, times are tough, Pam. Deal with it.

Jim: You cut more than you had to, didn't you?

Dwight: Sure.

Jim: Well, why did you do that? You work here, don't you want good insurance?

Dwight: Don't need it. Never been sick. Perfect immune system.

Jim: OK, well, if you've never been sick, then you don't have any antibodies.

Dwight: I don't need them. Superior genes. I'm a Schrute. And superior brain power. Through concentration, I can raise and lower my cholesterol at will.

Pam: Why would you want to raise your cholesterol?

Dwight: So I can lower it.
Oscar: He literally won't come out of his office.
Kevin: He's got to come out sometime. To go to the bathroom.

Angela: Kevin! That's inappropriate.
Oscar: Michael, can I talk to you?
Michael: Ah, uh, I would love to, but I am really busy. Rain check?

Meredith: Michael. Michael, please, can we talk to you about this memo?

Michael: Ah, what? Which memo?

Pam: Dwight's health care memo. I told you about it.

Michael: Is it a good plan?

Dwight: It's a great plan. It saves the company a fortune.

Oscar: It's like a pay decrease.

Pam: Michael, he made huge cuts.

Michael: Cuts? What? Wow, Dwight, did you make cuts?

Dwight: Yeah, you said...

Michael: No, no , no, you know what? I said nothing specific because I was so busy. Why don't you go in there and find these people a plan that will work for them? OK?

Dwight: I can handle that.

Michael: OK? All right. Do we feel good? All right. Good. Plus, there's some other good news. Today, at the end of the day, I will have, for all of you, a big surprise. OK? So hang in there, and I will see you at the end of the day. Right?
Oscar: This is not good.
Angela: It's ridiculous. Did you talk to him?

Oscar: What was that?

Angela: You let him walk all over you. It's just pathetic.

Kevin: What are you guys talking about?

Angela: Nothing, Kevin.
Michael: Do I know what the surprise is? Hell no! It doesn't matter. The point is, they're not unhappy anymore. They're out there thinking, "Wow, my boss really cares about me. He has a surprise. He's cool. I... what a great guy. I love him. I... love him.
Dwight: OK, everyone. Gather round. Step forward. It has been brought to my attention that some of you are unhappy with my plan. So what I'd like you to do is to fill this out and write down any diseases you have that you might want covered and I'll see what I can do.
Jim: OK, you know what Dwight? We can't write our diseases down for you because that's confidential.

Dwight: OK, well, I didn't say to write your name down, did I? Fill it out, leave it anonymous. Or, don't write any disease down at all and it won't be covered. Sound fair? Good. I'll be in my office.

Jim: Workspace.
Michael: You know what? Come with me. We are going on a little mission. Operation Surprise.
Pam: Where are you going?

Michael: Um, headed out. Part of my busy day, you know. Meetings. [Giggles] Couldn't find the knob.
Michael: So, basically, I want to do something nice for my employees. Atlantic City, OK? They have this thing where they send a bus, right, for free. Picks everybody up, you head down there, get to the hotel, room is comped, they give you a pile of chips, and your food, everything just kind of all-inclusive, free kind of weekend.
Travel Agent: I don't know of anything like that, but, um, you know what you might want to do, is just call those casinos directly. Um, maybe?

Michael: Yeah, yeah, yeah. I did, so...
Jim: Wait. What are you writing? Don't write Ebola or mad cow disease. Right? 'Cause I'm suffering from both.
Pam: I'm inventing new diseases.

Jim: Oh, great.

Pam: So, let's say my teeth turn to liquid and then, they drip down the back of my throat. What would you call that?

Jim: I thought you said you were inventing diseases? That's spontaneous dental hydroplosion.

Pam: Nice.

Jim: Thank you.
Michael: [on his cell phone] Calling you to ask you a little favoroonie my friend. Um, trying to give the troops around here a little bit of a boost. And I was thinking that maybe we could take them down to take a spin on your big ride.
Man on Phone: You mean the elevator that takes you down into the mineshaft? It's not really a ride.

Michael: Its says here that it's a 300ft drop.

Man on Phone: It goes 300 feet into the earth, but it moves really slowly.

Michael: So it's not a free fall?

Man on Phone: It's an industrial coal elevator.

Michael: Uh, all right. Well, once you get down into the mine, what... you got laser tag or something?
Michael: OK, so I don't know what the surprise is. Am I worried? No. No way. See, I thrive on this. This is my world. This is improv. This is Whose Line is it Anyway?
Dwight: Damnit! Damnit Jim!
Dwight: All right, who did this? I'm not mad. I just want to know who did it so I can punish them.

Jim: What are you talking about?

Dwight: Uh, someone forged, uh, medical information and that is a felony.

Jim: OK, whoa. 'Cause that is a pretty intense accusation. How do you know that they're fake?

Dwight: Uh, leprosy? Flesh eating bacteria. Hot-dog fingers. Government-created k*ll nanorobot infection.

Dwight: You did this, didn't you?

Jim: Absolutely not.

Dwight: Yes you did.

Jim: No I didn't.

Dwight: I know it was you. Fine. You know what? I'll have to interview each and every one of you until the perpetrator makes him or herself known. And until that time, there will be no health care coverage for any one!
Jim: k*ll nanorobots?
Pam: It's an epidemic.
Dwight: The problem, Jim, is that people who are really suffering from a medical condition won't receive the care they need, because someone in this office is coming up with all this ridiculous stuff. [reads off of paper] "Count Choculitis"
Jim: Sounds tough.

Dwight: Why did you write that down Jim? Is it because you know I love Count Chocula?

Jim: Do you?

Dwight: I think you need to confess...

Jim: Mmm hmm.

Dwight: ...the fact...

Jim: Yep.

Dwight: What are you doing? Those are my keys.

Jim: Good luck. [closes door and locks it]

Dwight: Jim! Damnit! No! Jim! Let me out! Jim! Let... [Without looking, Jim throws his keys to his left, they land on a shelf next to Stanley]

Stanley: [looks at keys, continues talking on phone] ...the light green or green...
Jim: [answering phone] Jim Halpert.
Dwight: Let me out.

Jim: Who is this?

Dwight: Let me out or you're fired.

Jim: No, you can't fire me.

Dwight: Yes I can. I'm manager for the day. Clean out your desk.

Jim: OK, can you hold on? I'm getting the, ah, beep. [presses button on phone]] Jim Halpert.

Pam: [on phone] Hey, Jim. It's Pam.

Jim: Hey Pam! How are you?

Dwight: Jim! Open the door!

Pam: Good, how are you? Busy?

Jim: I'm doing OK. Getting excited for the weekend though. What are you up to?

Dwight: Jim!

Pam: Um, I'm not bothering you, am I?

Jim: No, not at all.

Pam: You don't have anything you're doing?

Jim: I have nothing to do.

Dwight: Jim!

Pam: Oh great. Um, no, this weekend? Nothing. I'm not really doing anything.

Dwight: Jim!

Jim: Oh yeah?

Pam: I might go to the mall.

Jim: The mall?

Dwight: Jim!

Pam: I need new shoes.

Jim: Oh, interesting, what kind of shoes?
Jan: Hello?
Dwight: Uh, hello. Uh, this is Dwight Schrute calling for Jan Levenson-Gould.

Jan: This is Jan.

Dwight: Hi. Dwight Schrute calling, acting manager, Scranton branch. Listen, I needed your permission to fire Jim Halpert.

Jan: Who is this?

Dwight: Dwight Schrute.

Jan: From sales?

Dwight: Well...

Jan: Where's Michael Scott?

Dwight: He is not here right now. He put me in charge of the office.

Jan: Dwight, listen to me very carefully. You are not a manager of anything. Understand?

Dwight: That's not entirely true, because he put me in charge of picking the health care plan.

Jan: Really?

Dwight: Yeah.

Jan: OK, when Michael gets back, you tell him to call me immediately.

Dwight: Call you immediately. Good. Oh, hey, listen, um, since I have you on the phone, um, can I fire Jim?

Jan: No. Please don't use my cell phone ever again.

Dwight: Oh, this is your cell, I thought this was your... [dial tone]
Michael: Hey, hey, everybody, Ice-cream sandwiches! Aaaahh! [laughs] Here you go. Take one, take one. It's all good. Phyllis, think fast. Ya-bome! Oh, oh, I see Angela. Angela? Right? Waaaaah! Oh, hey temp. Why don't you take two? Because you don't get health care. And uh, faster metabolism.
Ryan: Did you get the kind with the cookies? Instead of the...

Michael: Why don't you just eat it, OK? And here you go, Stanley the manly.

Stanley: Oh, thanks.

Michael: There you go.

Stanley: This isn't the big surprise, is it? Because we've been having a pretty horrible day.

Michael: Uh, nope. Nope. This isn't the surprise. It's surprising, um... because you didn't expect it. But you will... you'll know it when you see it.

Dwight: Michael. Michael?

Michael: [under his breath] Oh, Christ.
Dwight: I tried being rational, OK? And what happened? The employees went crazy, I got no help from corporate. That leaves me with no options.
Dwight: I'm now going to read out loud your submitted medical conditions. When you hear yours read, please raise your hand to indicate that it is real. If you do not raise your hand, it will not be covered.
Stanley: What about confidentiality?

Dwight: You know what? You have forfeited that privilege. I have tried to treat you all as adults, but obviously I am the only adult here. Number one, inverted penis.

Meredith: Could you mean vagina? Because if you do, I want that covered.

Dwight: I thought your vagina was removed during your hysterectomy?

Meredith: A uterus is different from a vagina. I still have a vagina.
Dwight: OK, great. Dermatitis. Thank you Angela. I'll make sure that's covered. OK, now. Who wrote this, hysterical one? a**l fissures?
Kevin: That's a real thing.

Dwight: Yeah, but no one here has it.

Kevin: Someone has it.
Kevin: Do you think we should go ?
Oscar: I don't know, Kevin. This is important. I don't want... [spots Michael through the blinds] There he is.

Kevin: What is he doing?

Oscar: I don't know.
Oscar: Well?
Michael: Well, what? You could be referring to anything.

Oscar: OK, the health care plan.

Pam: Why did you put Dwight in charge of that? He did a horrible job.

Michael: Uh, Dwight? Did you raise benefits?

Dwight: I most certainly did not.

Michael: Oh come on! That's horrible! Aaah... Thanks, Dwight, for a crappy plan. Ah, Damn! Oh, mmm, I wish I had time to change it, but Jan needs it by five, and... what time is it, what time is it? [looks at watch] Ah, it's after five. Oh, oh it's awful. So, well, OK. See you guys on Monday.

Angela: What about the surprise?

Michael: Oh... Yes. Exactly. Thank you Angela, for reminding me. Terrific. Um, before I tell everybody what the big surprise is, would you like to tell me what you think the big surprise is?

Stanley: We all think you don't have a surprise.

Michael: All right, I have some news for you. There is a big surprise. And... here it is. Here we go. And the big surprise is... Brrrrrrrr! Drum roll... Brrrrrrrr! Brrrrrrrr!
Michael: When I am backed into a corner, that is when I come alive. See I learned improve from the greats, like, um, Drew Carey and Ryan Stiles.
Michael: [clapping hands] God, yeah... Ah! This...
Michael: Robin Williams. Oh, man, would I love to go head-to-head with him. Oh! That would be exciting. [as Robin Williams] "Hi. I'm Mork from Ork." Well, I'm Bork from Spork. Nanoo, nanoo. Jibelee, baloobaloo.
Dwight: Oh, um... Jan wants you to call her.
Dwight: Michael!?
Michael: Oh! God. Dwight, come on...

Dwight: I wanted to talk to you about the downsizing?

Michael: There's no downsizing.

Dwight: I, but if there were, I'd be protected as assistant regional manager?

Michael: Assistant to the regional manager Dwight.

Dwight: Yeah, so I don't have to worry?

Michael: Look, look, look. I talked to corporate, about protecting the sales staff. And they said they couldn't guarantee it if there's downsizing, okay?

Michael: But there's no downsizing, so just don't...

Dwight: Bottom line. Do I need to be worried?

Michael: Mmm, mm, mm. Maybe.
Michael: It looks like there's gonna be downsizing. And it's part of my job, but... blah! I hate it. I think the main difference between me and Donald Trump is that, uh, I get no pleasure out of saying the words, "You're fired." [as Donald Trump] "You're foir-ed. Uh, you're foir-ed." It just makes people sad, and an office can't function that way. No way. [as Donald Trump] "You're foir-ed." I think if I had a catchphrase it would be, "You're hired, and you can work here as long as you want." But that's unrealistic, so...
Dwight: It's a real shame, 'cause studies have shown that more information gets passed through water-cooler gossip than through official memos. Which puts me at a disadvantage, because I bring my own water to work.
Stanley: Why'd you do this?
Dwight: I didn't do it. What do you mean? Oh, the water cooler was brought over here for... maintenance. So what do you guys hear? What's the scuttlebutt?
Michael: Get set for Operation Morale Improvement starring Michael Scott. Now, I think I have had a little stroke of genius in that I have had my assistant Pam... Smile, Pam. I have had her go out and find out whose birthday is coming up, so we can have a little celebration for it. Not bad, not bad at all. All right. And the birthday person is... drum roll please. Here we go, who is the birthday, birthday person?
Michael: Who is it? Who's the birthday?

Pam: Um... Actually, we don't have any staff birthdays coming up.

Michael: Next person on the...

Pam: Oh.

Michael: ...calendar.

Pam: Okay, umm... that would be Meredith.

Michael: Yes! All right, come on down Meredith!

Pam: But it's not until next month.

Michael: Um... uh, OK. Well, great, well, you know, it'll be a surprise.

Pam: You still want to have a party?

Michael: Yeah, why not? Sure. Go ahead, live a little. Come on, Pam. Come on, shake it up. Shake it up! Shake it up!

Michael: [grabs cell phone off desk] Brrrp! Uh, Spock, are there any signs of life down there? Well, let me check Captain. Eeee. Eeee. Eeee. Eeee. No, Captain. No signs of life down here. Just a wet blanket named Pam. Brr-rrrp. Star Trek.
Phyllis: Well, uh, for decorations, maybe we could... it's stupid, forget it.
Angela: What?

Phyllis: I was just going to say, maybe we could have streamers, but that's dumb, everybody has streamers. Never mind.

Angela: No, yeah, I think that's a good idea.

Phyllis: Yeah?

Angela: What color do you guys think?

Phyllis: Well, there's green, um, blue... yellow... red...

Pam: How about green?

Angela: I think green is kind of whoreish.
Pam: This was tough. I suggested we flip a coin. But Angela said she doesn't like to gamble. Of course by saying that she was gambling that I wouldn't smack her.
Michael: These are my party-planning beeyatches. Pulled off an amazing '80s party last year. Off the hook!
Michael: So I was thinking, if you haven't already got a cake, um, maybe going for one of those ice-cream cakes from Baskin-Robbins. Those are very good. Very Delicious.

Angela: Meredith's allergic to dairy, so...

Michael: She's not the only one that's going to be eating it, right? I think everybody likes ice-cream cake. It's not, uh, it's not just about her, so...

Pam: It is... her birthday.

Michael: Mint chocolate chip! That'd be good, how about some, mint chocolate chip?
Dwight: Hey, so listen, I was thinking that it might be a good idea if you and I formed an alliance. 'Cause of the downsizing? I think an alliance might be a good idea, you know. Help each other out.
Dwight: Do you want to form, an alliance, with me?

Jim: Absolutely, I do.

Dwight: Good, good. Excellent, OK. Now we need to figure out who's vulnerable and who's protected...
Jim: At that moment, I was so happy. I mean, everything Dwight does annoys me.
Dwight: Did you get your tickets?
Jim: To what?

Dwight: The g*n show. [Rolls up his sleeve and kisses his bicep]
Jim: And I spend hours thinking of ways to get back at him, but only in ways that could get me arrested. And then here he comes and he says "No, Jim, here's a way."
Dwight: There's one other thing and this is important. Let's keep this alliance totally a secret. Don't tell anyone.
Pam: An alliance?
Jim: Oh yeah.

Pam: What does that even mean?

Jim: I think it has something to do with Survivor, but I'm not sure.

Jim: Um, I know that it involves spying on people and we may build a fort, underground.

Dwight: Jim! Hey. Hi, Pam. Listen, could I talk to you a second about the... paper products?
Dwight: Did you tell Pam about the alliance?
Jim: What? No.

Dwight: Just now.

Jim: What? Oh no no no. Dwight, no. I'm using her, for the alliance. Who knows the most information about this office? Pam.

Dwight: Right, that's good, good, pursue this.

Jim: Well I'm trying to. Do you see what I'm doing?

Dwight: Mmm hmm.

Jim: But listen, I'm going to have to talk to her a lot. All right? And there may be chatting, and giggling. And you gotta just pretend to ignore it. Wipe it away.

Dwight: Done.

Jim: All right.
Michael: [to the camera] Can you get her? She's right there. [camera zooms in on Meredith at here desk] That is Meredith, the birthday girl. And this... is Meredith's card. Happy Bird-Day. [laughs] Um, let's see. Jim, Jim wrote, "Meredith, I heard you're turning 46, but, come on, you're an accountant. Just fudge the numbers." Not bad, pretty funny, I don't appreciate condoning corporate fraud though. Uh, here's the thing. Whatever I write here has to be really, really funny. Because people out there are expecting it. I've already set the bar really high. And they're all worried about their jobs, you know. It's kinda dark out there. Can you imagine if I wrote something like, uh, "Oh, Meredith. Happy Birthday. You're great. Love, Michael." [pretends to vomit and laughs]
Dwight: They seem awfully chummy, don't you think?
Jim: Yeah, what do you think that's about?

Dwight: Only one way to find out.

Jim: I'm on it.
Jim: You are not going to believe this.
Dwight: What? I believe it.

Jim: Well, tensions were high in the kitchen.

Dwight: I could tell, from the body language.
Jim: Hey Kev, that looks good. What is it? Turkey?
Kevin: Italian.

Jim: Oh, Italian. Nice. Wow! You got the works there. Red onion, provolone...

Kevin: Yeah.
Jim: Toby and Kevin, they're trying to get Angela kicked off.
Dwight: Good, let 'em. It helps our cause.

Jim: Well, I don't know, if Kevin's in accounting, and Toby's in Human Resources and they're talking...

Dwight: Oh, they're forming an alliance
Toby: I love their sandwiches.
Jim: I love their sandwiches too.

Kevin: Their bread's really good.

Jim: Their bread is very good.
Dwight: Damn it. God!
Jim: OK, listen, we need to assume that everyone in the office is forming an alliance and is therefore trying to get us kicked off.

Dwight: God... Damn it! Why us?

Jim: Because we're strong, Dwight. Because we're strong.
Michael: [staring at birthday card] Meredith, Meredith... Meri... Mary had a little lamb. Mary... Meredith had a little lamb. Don't bring that lamb to work or it'll poop on the floor.
Michael: Hey, Oscar! Come on in. What's up?

Oscar: Uh, I'm sorry to bother you.

Michael: Oh, not at all. Come on in. What's going on?

Oscar: My nephew is involved with, um, a charity for cerebral palsy, and I was wondering if maybe you'd like to... you know... if...

Michael: What?

Oscar: Donate to the charity?

Michael: Oh, God. Of course I would. Get it over here. Get that over here.

Oscar: Thank you.

Michael: No, I'm always good... for some serious buckage. Wow. Two dollars, three dollars? People out here do not care about diseases. I am going to give you... $25.

Oscar: That's... that's... that's very generous.

Michael: Oh, my gosh, well... Listen, Oscar, generosity and togetherness and community all convalescences into... morale. That's what I say, so...
Pam: [whispering] Hey, Jim, can I talk to you for a second?
Jim: Sure, what's up?

Pam: Um, I don't know, I'm just like, I'm going a little crazy 'cause I keep overhearing all these conversations between Michael and corporate about like, staff issues?

Jim: Oh no?

Pam: Yeah, he's making me take notes on these meetings and I'm, like, "These people are my friends." But he's all like, "This is confidential. You can't tell anybody." But I just feel like I want to... aaah. Just promise me you're not gonna say anything.

Jim: No, will not, I'm not going to tell anybody. This is between you and me.

Pam: OK, yeah.

Dwight: Jackpot.
Jim: That was beautiful. All her idea too. Awesome. She is so great.
Michael: [looking at birthday card] Meredith, bad breath. Meredith has bad breath.
Dwight: Hey, you wanted to see me?

Michael: Yeah. What do you know about Meredith?

Dwight: I don't think she'd be missed.

Michael: There's not going to be downsizing Dwight, OK? I just, I need to know a little bit more about my friend.

Dwight: Name, Meredith Palmer. Uh, personal information, divorced twice, two kids. Uh, Employer, Dunder Mifflin Paper Incorporated. Awards, multiple Dundies.

Michael: I know all that. I know all that. I just, I need something kind of embarrassing, you know. Kind of fun, inside.

Dwight: She had a hysterectomy.

Michael: [laughs] Which one is that again?

Dwight: That's where they remove the uterus.

Michael: Oh God! Dwight, no. I'm trying to write something funny. What am I going to do with a removed uterus?

Dwight: It could be kind of funny.

Michael: You know what, I am on a deadline here, and just... OK. Thanks, thanks for your help. I'll work it out. Thank you Dwight. That was a waste of time.
Jim: OK, here's the deal. All right? Pam says that one of the alliances is meeting in the warehouse during Meredith's birthday.
Dwight: Oh my God, we have to be there.

Jim: I know, but it's gonna be a little tough because there's no good place to hide there.

Dwight: No no, yes there is. Behind the shelves. Oh my God.

Jim: What? What?

Dwight: I know. I know exactly what to do.

Jim: [gives Dwight a high five] Great.
Dwight: I'm a deer hunter. I go all the time with my dad. One thing about deer, they have very good vision. One thing about me, I am better at hiding than they are... at vision.
Dwight: This is going to be perfect, OK? Centrally located. Perfect cover. I can hear and see everything.
Jim: Good.
Ryan: Michael? Are you done yet?
Michael: Almost there. Just a sec. Just a second. It is perfect, thank you. Excellent, here we go. It is time, thank you. OK, come on. Let's go! Get the cake. Here we go. Come on! Shhh. Be quiet.
Jim: Wait, this isn't gonna work. The lid's open.
Dwight: So tape it down.

Jim: I can't do that. You won't be able to breathe.

Dwight: Look, I can breathe just fine. OK, but if it makes you feel better, I'll poke holes in the box.

Jim: Thank you, thank you. OK.
Everybody: Surprise!
Meredith: Oh! Surprise.

Angela: No, it's ah...

Michael: It's surprise Meredith. One, two...

Everybody: [tunelessly] Happy birthday to you.

Michael: Find a key.

Everybody: Happy birthday...
Jim: So do you want me to stay here and, you know, stand next to the box?
Dwight: No, you need to go upstairs to the party so people don't notice we're both gone.

Jim: Right... That's good.
Dwight: Can I trust Jim? I don't know. Do I have a choice? No, frankly, I don't. Will I trust Jim? Yes. Should I trust Jim? You tell me.
Everybody: [singing] ... birthday, dear Meredith Happy birthday to you...
Michael: And many more!
Stanley: Last year, five years ago...
Michael: You were surprised, weren't you?

Meredith: Yes.

Michael: You looked freaked, man. We said "Surprise." You were, like, "What?" "What the hell's goin' on here?" Good cake. Why don't you have some?

Meredith: Uh, I can't. Um...

Michael: Come on. A little bit.

Meredith: I can't eat dairy.

Michael: Oh, right. God, too bad. It's so good.

Meredith: Yeah, it makes me sick.

Michael: You know what? If I were allergic to dairy, I think I'd k*ll myself. 'Cause this is way, way too good.
Pam: He's in a box?
Jim: Pam, he's in a box. He's downstairs, in a box, on the floor, near the shelves. I'm serious. Go down there and work your magic.
Pam: [on her cell phone] Hey where are you? Yeah, we were supposed to meet here. What? Oh my gosh! That ties in perfectly with something that Michael was telling me earlier! I just don't know what some of the people in, like, accounting are going to do? It said specifically that...
Dwight: [box falls over] Oh.
Michael: Jim, good party, huh? Just a little something I whipped up. You know, a little morale boost. No big deal.
Jim: Speaking of which, I meant to tell you. Very impressive, the uh, donation you gave to Oscar's charity. What was it? 25 bucks?

Michael: Well, you know, money isn't everything Jim. It's not the key to happiness. You know what is? Joy. You should remember that. Maybe you'll give more than three dollars next time.

Jim: Yeah, well, three dollars a mile. It's gonna end up being like 50 bucks. So... God, I can't even calculate what you're gonna have to give.

Michael: Is Oscar around?
Michael: I just thought it was kind of a flat, you know... 25 dollar, one-time donation. I didn't think it was per mile kinda deal. You know, so...
Oscar: Well, that's what a walk-a-thon is.

Michael: I know...

Oscar: It says it right on the sheet. Look, look at the sheet. It says, "However many dollars per mile."

Michael: Right. Got it. Yes. So it does. Um...

Oscar: I just think it's kind of cheap to un-donate money to a charity.

Michael: No, no, no, no, no. That wasn't what I wasn't, that wasn't... No. It-it-it's not about the money. It's just... it... it's the ethics of the thing, Oscar. How's your nephew? Is he in good shape?

Oscar: Yeah.

Michael: How many miles did he do last year?

Oscar: Last year, he walked 18 miles.

Michael: Son of a bitch. That is impressive.
Pam: Happy Birthday. [gives Meredith her card]
Michael: Read it out loud. And say who wrote everything so we know whose is the best.

Meredith: "Happy Bird-day" Um... "Meredith, good news. You're not actually a year older because you work here, where time stands still."

Michael: [under his breath] I don't know about that.

Meredith: That was Stanley. "Meredith, happy birthday, you're the best. Love, Pam."

Michael: [pretends to vomit] Huh! Thanks, downer.

Meredith: This is from Michael. "Meredith, let's hope the only downsizing that happens to you is that someone downsizes your age."

Michael: Because of the downsizing. Rumors. And because you're gettin' old.

Meredith: No, I... I get it. It's funny.

Michael: [laughs] You didn't get the joke. So, that's cool. That's, you know what? Actually... I have a bunch of these, good ones, that I didn't use. Um... Oh, where's that? Oh, OK, here's a good one. Um... "Hey Meredith, Liz Taylor called, she wants her age back and her divorces back." 'Cause Meredith's been divorced like, twice. Is that right?

Meredith: You're right. You're right. Yes.

Michael: Divorce. Um... OK, "Meredith is so old..."

Oscar: How old is she?

Michael: Everybody? If... could do it? "Meredith is so old..."

Everybody: How old is she?

Michael: "She's so old, she went into an antique store and they kept her."

Michael: That wasn't even mine. I got that off the Internet. Website. Um, don't get mad at me.

Oscar: Uh, nice party Michael.

Michael: This isn't my fault. Ladies, not your best effort. The streamers? I think we could have done better than that.

Angela: Phyllis wanted red, I didn't.

Phyllis: Oh, boy... You...

Michael: OK, we... all right. People, hold on, hold on. Just a second. OK, I think we're losing sight of what is really important here. And that is that we are... a group of people... who work together. I was... I really wasn't gonna flaunt this. I have made a very sizable donation to Oscar's nephew's... walkathon. $25.

Oscar: Per mile.

Michael: Per mile, yes.
Michael: When I retire, I... don't want to just disappear to an island somewhere. I wanna be the guy who gives everything back.
Michael: A check for the kids, and for the team.
Michael: I want it to be like... "Hey, who donated that hospital wing that is saving so many lives?" "Um, well, I don't, I don't know. It was anonymous." "Well, guess what, [whispering] that was Michael Scott." "But it was anonymous, how do you know?" "Because I'm him."
Oscar: Thank you, Michael.
Michael: Come here. [hugs Oscar and In a low voice] Don't cash that till Friday, OK?
Toby: Really? Today?
Ryan: Yeah.

Toby: Oh, Happy Birthday.

Ryan: Thanks.

Toby: Yeah, I could say something.

Ryan: No, don't. Don't do that.
Jim: OK, OK. I have something that totally tops the box.
Pam: Oh, tell me, tell me.

Jim: OK. I have just convinced Dwight that he needs to go to Stamford and... [Pam starts laughing]... spy on our other branch. No no no.

Jim: But before he does so, I told him that he should dye his hair to go undercover.

Pam: [laughing] That's perfect!

Jim: If we can get him to drive to Connecticut... and put peroxide in his hair...

Roy: [yelling] What the hell is this? What are you trying to cop a feel or something? Huh Halpert?

Jim: No, no, dude, no.

Pam: Hey, Hey!

Jim: No, dude, no, I was just, listen! Whoa.

Pam: Come on.

Jim: God, I don't even, I don't even know how to explain this. Uh, um... Dwight, uh, asked me to be in an alliance. And then um... um... we were... we've just been messing with him. Uh, because of the whole alliance thing. Um...

Pam: It's just office pranks.

Jim: It's stupid. It's, it's just office pranks.

Roy: [looking at Dwight] An alliance? What the hell is he talking about?

Dwight: I have absolutely no idea.

Roy: Come on.
Dwight: Do I feel bad about betraying Jim? Not at all. That's the game. Convince him we're in an alliance, get some information, throw him to the wolves.
Dwight: [With blonde hair] That's politics baby. Get what you can out of someone, then crush them. I think Jim might have learned a very valuable lesson.
Michael: [to Jim] Hey, you ready?
Michael: All right, all right, secret sign. Hey, Ryan. [Ryan holds up his bag] Very good. Excellent, excellent.

Dwight: Michael!
Michael: Today at lunchtime we're going to be playing the warehouse staff at a friendly little game of basketball. My idea. Last time I was down there, I noticed they'd put up a couple of hoops, and I play basketball every weekend. So I thought, "This might be kinda fun." And so I started messing around and... I'm sinking a few, you know. Swish, swish, swish. Nothing but net. And their jaws just dropped to the floor. African-Americans! So... you know, it's really just a good friendly game, a reason to get together.
Michael: Pam, Pam, thank you ma'am. Messages, please. Thank you.
Dwight: Michael, can I talk to you, please? Privately? In your office? I think I should be on the team.

Michael: No. And that's not me being mean, Dwight. That is based on your past behavior.

Dwight: Oh, please.

Michael: [to camera] When I let him come to my pick-up game...

Dwight: I apologized for that.

Michael: [to Dwight] I vouched for you.

Dwight: Michael, I...

Michael: I vouched for you in front of Todd Packer, Dwight. All right, here's what I'm going to do. The hand strikes and gives a flower. You are not going to play basketball. But I need somebody to come in and take over the holiday and weekend work calendar.

Dwight: I can handle that.

Michael: Good. Excellent, it'll be fun. Because corporate, uh, wants someone to be here on Saturday. And so we're going to have to have some people come in on the weekend, and I know nobody's gonna want to do it and I know everybody's gonna complain and bitch and I don't want to have to deal with that.

Dwight: And that's why you have an assistant regional manager.

Michael: Yes it is. Assistant to the regional manager.

Dwight: [to camera] Same thing.

Michael: No, it's not. It's lower, so...

Dwight: It's close.
Dwight: So we need someone to work this Saturday and I think that, that should be...Jim.
Jim: God, this is so sad. This is the smallest amount of power I've ever seen go to someone's head. Phyllis, can you believe this?

Phyllis: Keep me out of it.
Pam: My fiance has plans for us this Saturday. So I really hope that Dwight doesn't make me work. Maybe I should sleep with him? I'm kidding, kidding. Totally kidding.
Michael: All right, managing by walking around. This is our warehouse. Or, as I like to call it, the whorehouse. But don't you call it that, I've earned the right.
Ryan: Fine, don't worry about that.

Michael: And here we have "Mister Roger's Neighborhood." Come on over here. Hey, this is Ryan. He's temping upstairs.

Lonny: What's up?

Michael: And this is the foreman. Mista Ra-jahs.

Darryl: It's not my real name.

Michael: No, it's Darryl. Darryl is Mista Ra-jahs.

Ryan: Darryl Rogers?

Darryl: Darryl Philbin. Then Regis, then Rege, then Roger, then Mister Rogers.

Michael: [laughs] And that is Lonny. And this is Roy. Roy dates Pam. You know, the uh, the best looking one upstairs.

Ryan: Yeah, yeah.

Michael: You still getting it regular man? Huh? I mean, I can tell her it's part of the job! Rapport!
Pam: [on the phone] No, no, I know that the warranty's expired, but isn't it supposed to last longer than two years if it isn't defective? OK, fine, three years.
Jim: Pam gets a little down. Her toaster oven broke. Um, which she got at her engagement shower. Um, for a wedding that still has yet to be set... and that was three years ago.
Michael: So, um, one o'clock sharp and we've got a game on.
Darryl: We're loading at one.

Michael: Oh, I see, you're chickening out on me. You're bailing on me.

Darryl: No, we got a truck going out at 1:15. So, that's the busy time.

Michael: Oh, well, I'm glad that some time is a busy time because whenever I'm down here it doesn't seem too busy to me. Oh, oh. You can dish it out, but you can't take it. OK, fine, have it your way. [clucking and dancing like a chicken]

Darryl: All right, fine, you know what? One o'clock.

Michael: All right, see you at one.
Michael: Are we ready for the game?
Everybody: [half-heartedly] Yeah.

Michael: I... yeah, yeah. I know, grumble, grumble. But you would follow me to the ends of the earth, grumbling all the way. Like that, uh, dwarf from Lord of the Rings.

Dwight: Gimli.

Michael: Nerd. That is why you're not on the team.

Dwight: Just trying to be helpful.

Michael: Uh, [in a nerdy voice] "I'll help, Elwyn Dragonslayer, uh, ten points, power sword."

Jim: That's him.

Michael: OK, so, let's put together a starting line-up, shall we? Stanley of course.

Stanley: I'm sorry?

Michael: Um, what do you play? Center?

Stanley: Why "of course"?

Michael: Uh...

Stanley: What's that supposed to mean?

Michael: Uh, I don't know. I don't remember saying that.

Jim: Uh, I heard it.

Michael: Well, people hear a lot of things, man. Um... other starters... Me, of course. I heard it that time.

Phyllis: I'd like to play if it's just for fun. I played basketball in school.

Michael: [ignores Phyllis)] Um... Yeah. Who else? We have Jim. We have Ryan, the new guy, right? Untested. Willing to prove himself now. A lot of passion, a lot of heart.

Ryan: But, I'm getting paid to skip lunch?

Michael: Yes.

Ryan: OK.

Michael: Yes, this is business. The, uh, business of team building and morale boosting. Uh, who else?

Oscar: I can help out, if you need me.

Michael: I will use your talents come baseball season, my friend. Or if we box.

Kevin: I have a hoop in my driveway.

Michael: No.

Phyllis: I have a sports bra.

Michael: No, no, ridiculous.

Dwight: Michael, look. [Dwight throws paper at the garbage can] Missed it...

Michael: Close. All right, uh... Me, Stan the man, Jim, Ryan and Dwight.

Dwight: Yes!

Michael: Sorry Phyllis.

Dwight: Can I be team captain?

Michael: No, I'm team captain.

Dwight: Can I be team manager?

Michael: No, I am the team manager. You can be assistant to the team manager.

Dwight: Assistant team manager?

Michael: No.

Dwight: OK, we'll see who's working this weekend then.

Michael: Jim, you're in charge of the vacation schedule now.

Jim: Oh my God.

Michael: thr*at neutralized.
Michael: [hits Pam in the head with a piece of paper] Off the backboard!
Pam: Please don't throw garbage at me.

Michael: Oh, Pam with a zinger. Hey, Pam, how would you, like to be our cheerleader today? You know, some, ah, pigtails? A little, ah, halter top, you could tie that up. And you know, something a little, just, youthful, for a change. Just this once?

Pam: I don't think so Michael. Besides, I can't cheer against my fiance.

Jim: I'll do it. Wear a little flouncey skirt if you want, and...

Michael: Yeah, I bet you would. Just try not to be too gay on the court. And by gay I mean, um, you know, not in a h*m* way at all. I mean the uh, you know, like the bad-at-sports way. I think that goes without saying.

Pam: Maybe Angela would cheerlead.

Michael: Oh, yeah right.

Phyllis: I'll do it.

Michael: Oh, yuck, that's worse than you playing. ... 'Cause we need you as an alternate in case somebody gets hurt. That's where we need you. Blessed be those who sit and wait. You made it, suit up, you're on the team! All right, cool! Very good.
Michael: Oh-oh. Oh-oh. A spy from the warehouse. Trying to figure out our plays, huh, man?
Darryl: Just getting a tea bag.

Michael: Oh ho, oh, he's running. He's running. He's running, but he can't hide because you know what? One o'clock, you better bring your 'A' game. Because me, and my, posse guys are gonna be in your face. Right in your face!

Darryl: Why don't we make it more interesting? Loser buys dinner at Farley's.

Michael: Whoa-ho. I like the way you think. You know what, I'm gonna take that one step further. Loser, works, on Saturday.

Darryl: No, that's not as much fun. You know what?

Michael: What?

Darryl: You're on.

Michael: OK. Cool, you're on. [to Dwight] Don't screw this up.
Michael: [to camera] Classic beginner's mistake, eating before a game.
Angela: Has anyone seen the first-aid kit? [Dwight holds the kit up] How many times have I told you? I'm the safety officer, not you.
Jim: Basketball? It was kind of my thing in high school. And I'm, yeah, I'm looking forward to playing. You know, I think I'm gonna impress a few people in here.
Jim: You coming down?
Pam: Yeah, I'm just forwarding the phones.

Jim: You gonna wish me luck?

Pam: Yeah, you're gonna need it.

Jim: Whoa.

Jim: Is that trash talk from Pam?

Pam: [laughing] I'm just saying, Roy is very competitive.

Jim: Oh.

Pam: And he wants to take the WaveRunners to the lake this Saturday so...

Jim: Well, I'm going to the outlet mall on Saturday, so if you wanna save big on brand names and Roy has to work, which he will, because I'm also competitive, you should feel free to come along.

Pam: Um, I think I'm gonna be up at the lake.

Jim: I think I'll see you at the mall. Yeah.
Michael: Hey, there he is! Secret w*apon! All right, guys, come on, let's bring it in! Here we go! OK, listen, this is just going to be a friendly game, right? We are all on the same team here, the Dunder Mifflin team. Of course, if you b*at us, you're fired. That's a joke. OK, let's do it.
Jim: Have a good game man.

Roy: Yeah, you too. Should be fun.

Michael: All right, everybody stretch out a little bit. Stretch it. Full stretch. Ryan, you wanna stretch?

Ryan: I stretched before I came.

Michael: OK.
Michael: OK, Ryan, you have Darryl. I have Roy.
Jim: Really? I thought I'd take Roy.

Michael: Actually, I think Roy is their best player not Lonny. So, Dwight, you uh, have the East German gal. Uh, who else we got... Um...OK, all right, you guys.

Dwight: [taking off his shirt] OK, we'll be skins!

Michael: Aw, come on Dwight.

Dwight: What? Shirts on or off?

Michael: On. Just put it on.

Dwight: You sure?

Michael: Yes. Uh, Pam? You kind of have your foot in both camps, why don't you do the uh, jump ball OK?

Roy: Don't listen to him Pam. Trust me, tip it my way or you're sleeping in the car.

Michael: Stanley! What? You gotta be kidding me! !?! [Roy steals the ball, and goes for a lay up] Oh... Here we go! [Lonny sh**t and makes it] Who's on him? Somebody get him!

Teammates: Yeah!

Roy: That's what I'm talking about.

Michael: Yeah, yeah, yeah. Over here, over here. [Jim saves the ball from going out of bounds and passes to Michael] Here we go. Three! [sh**t and misses] Let's go to the zone! We're going to zone!

Dwight: De-fense! [clap, clap] [Michael joins in] De-fense! [clap clap]

Michael and Dwight: De-fense! De-fense!

Warehouse worker: [Roy scores] Well done team.

Michael: Who's got Roy? [Jim does a behind the back move around Roy for the basket]

Pam: Woo!
Michael: [misses a half court sh*t] Aw, come on! What is wrong with me today!? Usually hit those. [Dwight scores] Dwight, I was open. All right, let's go.
Michael: [Roy bumps Michael to get around him] OK, foul. Charging. Charging. That's a foul.
Roy: OK.

Michael: OK, I'll take it. [misses free throw] OK.
Michael: When I am playing hoops all of the stress and responsibility of my job here just melts away. It's gone, I'm in the zone.
Michael: [misses another sh*t] What is wrong with me today?!
Michael: Who am I? Am I Michael Scott? I don't know... I might just be a basketball machine. What's Dunder Mifflin? I've never heard of it. Filing? Paperwork? Who cares? Possible downsizing? Um... well, that's probably gonna happen, actually.
Michael: Jim! Jim! Jim, right here, Jim! Give me the ball! Ryan, cut! [Michael looks away and misses Jim's pass] Whoa!
Jim: My bad.

Darryl: [scores] Here we go. Here we go. Here we go.

Lonny: [dancing] Where you at? Where you at? You over there? I'm over there.

Michael: That is cool. Is that like the Robot?
Michael: [Ryan scores] Nice! Come here! [gives Ryan a chest bump]
Ryan: Can we just do one? That's cool, that's fine.

Darryl: You have one more free throw sh**t. Come on.

Roy: All right, let's go.
Warehouse worker: Watch your back Madge.
Madge: Hey! Come on man!

Michael: Come on! Hey, Dwight. Dwight!

Dwight: [scores] Yeah! [points to Madge] In your face!

Madge: Yeah, like that counts.

Michael: You know what? Dwight, Dwight...
Michael: Football is like rock and roll, it's just bam-bam-boo... And basketball is like jazz, you know? You're kind of... Dupee-doo, dupee-do. It's all b*at, it's in the pocket, it's like... [singing] Dupee-do, dupee-do, dapee-dah...
Michael: [singing] Du-du-du-du-dupee-do, de-do-do-do. Du-du-du-du-dupee-do, de-do-do-do. Harlem Globetrotter...
Roy: [steals the ball, scores, mimics singing] Du-du-du-du-dupee-do. Your ball.

Michael: All right, time, time out. Come on, sales, over here. Bring it in! Come on!

Michael: What's going on? What's going on? You're playing like a bunch of girls.

Jim: You know what? Let me take Roy.

Michael: All right, switch. Take it up a notch, come on.
Michael: sh**t, sh**t it. [Roy hits Jim in the mouth with his elbow] Whoa, whoa, whoa, whoa! Foul! Naked aggression! Oh, that is... You all right Jim? Suck it up.
Darryl: Block, block, block!
Madge: He's afraid of you now.

Michael: [Jim makes a sh*t after pushing off Roy] Ouch! Oh, how much does it hurt? How much does it hurt?
Michael: [Jim pushes Roy to the ground and makes another sh*t] Yes!
Roy: What the hell man?

Jim: Take it easy.

Roy: No, you take it easy.
Michael: [Darryl scores] Watch the long passes, you guys!
Ryan: [Dwight steals the ball from Ryan] Same team, Dwight.
Michael: Dwight!

Dwight: [scores] Yes!
Michael: [Phyllis scores] Yeah! Yeah! Yeah! In, your, face! Angela, what's the score?
Angela: You're ahead.

Michael: Yeah, baby, here we go!

Michael: [Jim has the ball] Jim! Jim! Right here! [runs into the elbow of the guy guarding him] Ow! God! Hold it!

Worker: I'm sorry.

Michael: Foul! Foul!

Worker: I'm sorry. You all right?

Michael: Oh, that hurts.

Worker: Sorry, I didn't mean to do that.

Michael: What's your problem man? Gah, just clocking me for no reason?

Darryl: Take your sh*t man!

Michael: No, no, no, no. That was a flagrant, personal, intentional foul. Right there.

Worker: No it wasn't.

Michael: [mocking voice] Yes, it was. You know what, I'm just being fair.

Worker: Oh, really? No, I just put my arm up...

Michael: Game over. Game over. That is it! I'm sorry, you know? I hate to do it this way but, you know, that's just... we're having a friendly game. It's a shame. This is a damn shame, but we're like a family here and that just, that won't fly.

Angela: This is a cold pack...

Dwight: Here, give me that. You have to break the interior bag. [bag explodes]

Michael: Thanks Dwight.

Lonny: Wait, what does that mean? What is it, a tie? What's going on?

Michael: Well, let's just say whoever was ahead won.

Darryl: That was you.

Michael: It was us? Really? I didn't, I didn't know. Great, I mean, I guess you guys are working Saturday. Your face.

Roy: No, no, no, I'm not coming in on Saturday.

Darryl: Yeah, this isn't happening.

Michael: Um... well, you guys, you know, I'm the boss so...

Lonny: So what's that? We're coming in on Monday, right?

Michael: Hey, hey...

Lonny: Monday?

Michael: [laughing] You guys believed me? Come on. Dogs, you know, you should know me better than that. No, oh, do you think that would've been good for morale? No. No. No. Exactly, no. I'm embarrassed it was even that close though. So... nah, of course, we're coming in Saturday. Good game. Word.
Jim: [to Pam] ...so I talked to the scout, it looks good.
Pam: Mmm-hmm.

Jim: I didn't sign anything.

Roy: Hey baby.

Pam: Hey.

Roy: [to Jim] Look at Larry Bird. Larry Legend.

Pam: Yeah, he's, uh, pretty good, huh? [to Roy] Let's get you into a tub.

Roy: Yeah? Let's get you into a tub.
Michael: Hey, what a game, huh? What a game.
Oscar: What time do we have to come in?

Michael: Come on. Let's not be gloomy here man. We're all in this together. We're a team. You know what? Screw corporate, nobody's coming in tomorrow. You have the day off. Like coming in an extra day is gonna prevent us from being downsized. Have a good weekend.
Michael: The great thing about sports is that it is all about character. And you can learn lessons about life even if you don't win. But we did because we were ahead.
Jan: Are you listening to me Michael?
Michael: Affirmative.

Jan: What did I just say?

Michael: You just said, let me uh... check my notes. You just said...

Jan: Alan and I have created an incentive program to increase sales.

Michael: Hey, hey how is Alan? Tell Alan that the Mets suck! Okay? From me, big time. Go Pirates!

Jan: I'm not going to do that Michael.

Michael: Okay

Jan: We've created an incentive program to increase sales.

Michael: Uh, huh.

Jan: At the end of the month you can reward your top seller with a prize worth up to a thousand dollars.

Michael: Whoa. Howdy-ho. Wow, a thousand big ones. That's cool. Do I uh, do I get to pick the prize?

Jan: Uh, yes. Yes you can.

Michael: Um, question: Does top salesman include uh, people who were at one time such outstanding salesman that've been promoted to...

Jan: No, Michael. No. You can't win this prize.

Michael: I didn't mean me!
Michael: Well, first what we have to do is find out what motivates people more than anything else.
Dwight: Sex.

Michael: It's illegal. Can't do that. Next best thing.

Dwight: t*rture.

Michael: Tah, come on Dwight. Just help me out here. That's just stupid.

Pam: Uh, Michael?

Michael: Pam!

Pam: Hey, there's a...

Michael: Burger with cheese!

Pam: There's a person here...

Michael: And fries!

Pam: There's...

Michael: And shake! What? Go ahead.

Pam: There's a person here who wants to sell handbags.

Michael: No, no, no. No vendors in the office. That is a distraction.

Pam: Okay, I told her you'd talk to her.

Michael: Pam. Pam. Come on, I'm busy. So just tell her to go away.

Pam: Okay.

Michael: [exhales loudly, looks out window and sees Katy] Oooh, alright I'll talk to her.
Katy: This one is hand embroidered.
Michael: All right girls break it up, you're being infiltrated. Cock in the henhouse.

Dwight: Cocks in the henhouse.

Michael: Don't say cocks. Oh, what is your name, my fair lass?

Katy: Katy.

Michael: Ah, Katy. Wow. Look at you. You are, uh you're like the new and improved Pam. Pam 6.0.

[Pam looks embarassed at Michael - Katy looks sympathetically at Pam]

Michael: Oh, look. Oh hey, no catfights you two. I'm against violence in the workplace.

Dwight: So am I.

Michael: Nobody cares what you think.

Dwight: Doesn't matter.

Michael: So uh, you know what? I usually don't allow solicitors in the office but today I am going to break some rules, and you can have the conference room. It's yours. All day.

Katy: Wow, thanks.

Pam: There's an HR meeting in there at 11:30.

Michael: Well, lets put 'em in the hallway. Give 'em some chairs. Right? Decisiveness. One of the keys to success according to Small Businessman.
Michael: I do. I read Small Business man. I also uh, subscribe to USA Today and American Way Magazine, that's the in-flight magazine. Some great articles in that. They did this great profile last month of Doris Roberts and where she likes to eat when she's in Phoenix. Illuminating.
Michael: This is my conference room. So please, uh, make yourself at home. Whatever you need, I'm right on the other side of this wall. [knocks on wall] used to be a window here. There's not anymore. So, that's where I will be.
[Katy unpacks her handbags]

Michael: So if you need anything else, something to make you more confortable just don't hesitate to ask. I'm right here.

Katy: I guess a cup of coffee would be great.

Michael: Wait a second. I should have spotted another addict. Uh, gotta love the 'bucks.

Katy: What?

Michael: It's like a slang for Starbucks. They're all over the place. Oh, man, that place is like the promised land to me. What a business model too. Ah, too bad we don't have the good stuff here.

Katy: Regular coffee is fine.

Michael: Nah, it's not. it's spppplllibbb

Katy: No really it is.

Michael: No, here's the thing. Y'know I do my best to be my own man and go by the b*at of a different drummer and nobody gets me, and they're always putting up walls and I'm always tearing 'em down, just breakin' down barriers, that's what I do all day. So a coffee, regular coffee for you. High test, or unleaded?

Katy: Bring it on.

Michael: Oh. Woo, I will. I will bring it on. Ah, all right.
Kevin: So are you jealous 'cause there's another girl around?
Pam: No.

Kevin: She's prettier than you though.

Pam: That's a very rude thing to say, Kevin.

Kevin: [nods]
Katy: So do you like the periwinkle and the purples?
Dwight: The purse girl hits everything on my checklist. Creamy skin. Straight teeth. Curly hair. Amazing breasts. Not for me, for my children. The Schrutes produce very thirsty babies.
Michael: [handing Katy a mug of coffee] There ya go. Nice steaming cup o'joe.
Katy: Thank you.

Michael: I have an idea. Why don't I introduce you around, you know you can kind of get your foot in the door, meet potential clientele, right?

Katy: Gosh, I would love to but, my purses, I should, um...

Michael: Oh, um, well, we could have Ryan take a look. Ryan, would you look after the purses, please?

Ryan: I'm installing File Share on all the computers.

Michael: Yeah, well, bladdy-bluda-blah-blah. Techno-babble. Just do it, okay. We have company. Right?
Michael: You should sell a lot here because this branch made over a million dollars last year. Not that we're all millionaires. I'm probably closest. So here's Oscar. Oscar, this is Katy.
Oscar: I'm on the phone.

Michael: Oooh-ooh. Oscar the grouch. Right? I thought of that.

Katy: That was on Sesame Street.

Michael: I know. I know. I made the connection. Can you believe he'd never heard that before he worked here?

Katy: No, I don't believe that.

Michael: I know, it's unbelievable.
Pam: It's nice having Katy around. It's another person for Michael to um, interact with.
Michael: Here is Toby from Human Resources. Katy, Toby.
Katy: Hi

Toby: Hi, nice to meet you.

Michael: Toby, Katy.

Toby: Hey, um did you go to uh, Bishop O'Hara?

Katy: Yeah.

Toby: Yeah, me too.

Katy: Cool. What year were you there?

Toby: Eighty-nine.

Michael: Toby's divorced. He uh, guh recently, right?

Toby: Yeah.

Michael: You and your wife, and you have kids.

Toby: A girl.

Michael: Oh that so - that was really messy. He slept one night in your car too?

Toby: [looks resigned]

Katy: I should probably get back to my table.

Michael: Okay. Alright. Cool. See ya in a bit. [looks at picture on Toby's desk] Oh, she's cute. Cutie-pie. Back to work.
Michael: I live by one rule. No office romances. No way. Very messy. Inappropriate. No. But, I live by another rule: Just do it. Nike.
Roy: Hey, Jimmy what do you think of that little purse girl, huh?
Jim: Cute, sure, yeah.

Roy: Why don't you get on that?

Jim: She's not really my type.

Roy: What are you gay?

Jim: Hmmm, I don't think so. Nope.

Kevin: What is your type?

Jim: [glances at Pam] Moms, primarily. Yep. Soccer moms. Single moms. NASCAR moms. Any type of moms, really.

Roy: That's disgusting.

Kevin: Stay away from my mom.

Jim: Too late, Kev.

Roy: [Katy walks through breakroom] Man, I would be all over that if I wasn't dating Pam.

Pam: We're not dating, we're engaged.

Roy: Engaged, yeah.
Jim: Pam and I are good buddies. I'm sort of Pam's go-to guy for her problems. You know with stuff like work, or uh, her fiance Roy. Or uh... Nope, those are pretty much her only two problems.
Jim: She'd be perfect for you.
Dwight: Hmmm... she's been talking to Michael a lot.

Jim: So, what? You're Assistant Regional Manager.

Dwight: Assistant to the Regional Manager.

Jim: Well, you know what Dwight? He's your work boss, okay? He is not your relationship boss.

Dwight: That's true.

Jim: Plus you have so much more to talk to this girl about, You're both um, salesmen. I mean that's something right there.

Dwight: True. Plus I can talk to her about the origins of my last name.

Jim: It's all gold.
Katy: Guys are usually my best customers, they buy the high end stuff like the beads and the sequins and stuff. For gifts, you know? They don't know what they are looking at. So I make suggestions.
Jim: Alright. Here's the thing okay, you just keep talking to her. If you hit a stall you have a perfect fall back.
Dwight: What's that?

Jim: You buy a purse.

Dwight: I don't want a purse. Purses are for girls.

Jim: Dwight, that's not necessarily true. Do you read GQ?

Dwight: No.

Jim: Okay, I do. There like mini briefcases, alright? Lots of guys have them.

Dwight: Like those?

Jim: Yes. Listen, you are spending way too much time talking to me, when you could be talking to her.

Dwight: Okay, I'm just going to use the bathroom, and then I'm going...

Jim: No. You don't need the bathroom. You've got it. Go.
Jim: Okay, shhhh stop... stop whatever you're doing because this is going to be good.
Pam: [smiles]

Jim: [mimicing Dwight in high-falsetto voice] Hi my name's Dwight Schrute and I would like to buy a purse from you. Good lord, look at these purses! This is something special. Oh my God is this Salvatore Di-chini-asta?

Pam: [mimicing Katy] Oh definitely, definitely step in and out of it like that.

Jim: Yes, well I want to stress test it. You know, in case anything happens.

Pam: Oh!

Jim: Oh! That was really. [Dwight hits purse against table] This is necessary to do to really give it a good workout. This is the ooooh... This is the prettiest one of all.

Pam: Oh...

Jim: I'm going to be the prettiest girl in the ball. Oh, how much?

Pam: Oh, God. It's sad. It's so sad.

Jim: [whispering] Here he comes, shhh...

Jim: [gives Dwight a thumbs-up - mouths the word] Good.

Pam: [smiles in agreement]

Jim: He did pick a good one.

Pam: You're horrible.
Katy: This one's really good for a hot date.
Pam: Yeah, what's that?

Katy: [laughs]

Pam: I'm engaged. So...

Katy: Congratulations. You need a hot date more than anyone.

Pam: I wished, right?

Michael: Giggle-giggle, juji-juji, I get it, I get it. Divine Secrets of the Ya Ya Sisterhood over here right? [to Katy] So how's that uh, coffee from earlier?

Katy: Good.

Michael: Ah, I knew it. Guzzled it down. You greedy little thing. So, uh, Pam is this your lunch break, or was that earlier when you were eating in the kitchen with those guys?

[Pam sheepishly hands Katy the purse and leaves]

Katy: [whispers] Sorry.

Michael: Busted.

Katy: [to Pam] Come back...

Michael: Oh hey, I want to show you something. Come here I want to show you something. I know you are going to like this. Picked it up today. A thousand big ones.

Katy: Is that from Starbucks?

Michael: Yes. This is a Starbucks digital barista. This is the mack daddy of espresso makers.

Katy: Wow. Is that for the office?

Michael: Oh, I know what you're thinking. You're not prying this out of my hands, but don't tempt me because I'll give it to you!

Katy: I wouldn't think of it.
Michael: Coffee is the great incentivizer in the office. It's a drug. It is quite literally a drug that speeds people up. It's not the only drug that speeds people up. You hear stories about Dunder Mifflin in the eighties before everybody knew how bad cocaine was. Guh. Man, did they move paper!
Michael: [Katy reading text message on her phone] Oh the rotating um, steam wand. [Katy looks annoyed] What? What's the matter?
Katy: Oh, nothing. My ride just bailed on me.

Michael: Oh, oh! God. I'm sorry. Is there...?

Katy: Oh no, it's um...

Michael: Where you going? Nearby? Because I can give you a ride.

Katy: No...

Michael: Seriously. No, really.

Katy: No. I really don't want to inconvenience you.

Michael: God! No, no, no, no. No inconvenience. I mean I'm out of here at five sharp.

Katy: At five?

Michael: I can go earlier. 'Cause I'm the boss. You know, whatever. I'm out of here slaves.

Katy: Okay.

Michael: What?

Katy: Okay, I guess that would be, I guess that would be okay.

Michael: Okay. Sounds good. Sounds good. Five o'clock sharp. I will give you and your purses a ride home.

Katy: Okay. Cool.

Michael: Excellent.

Katy: Cool.

Michael: Great. Cool. Cool. [takes deep breath - looks at camera] Yeah, okay.
Michael: I should have never let the Temp touch this thing. I had all these great icons and now I have four folders. So..
Dwight: It's actually better this way.

Michael: No it's not. Because I could just click on the icon and then I'm onto---

Dwight: Michael could I ask you something? I wanted to ask your permission to ask out Katy. I know it's against the rules and everything. Because...

Michael: No, no, no it's not against the rules. She's not a permanent employee so it's not.

Dwight: Thank you, Michael. I appreciate this so much.

Michael: But I think you should just know that I am going to be giving her a ride home later.

Dwight: What?

Michael: She asked me for a ride and so I am going to give her a ride home.

Dwight: Is that all it is? Just a ride home? Like a taxicab?

Michael: Well, might be a ride home. Might be a ride home and we stop for coffee and dot-dot-dot...

Dwight: Please. Please, I am your inferior and I'm asking you this favor. Can you promise me that it will just be a ride home?

Michael: No. I cannot promise you that.

Dwight: You cannot promise me, or you won't promise me?

Michael: Listen, Dwight.

Dwight: Do you love her?

Michael: [laughs] Dwight, no. I don't know. It's too early to tell. I don't know how I feel. [Dwight sadly looks away]
Katy: I think you've made a really good choice, she's really going to like that.
Stanley: Hmmm...

Michael: Espresso?

Katy: Oh, thank you.

Michael: You're welcome. Thank you. Hmmm-hmm-hmm.

Stanley: Is that from the machine that was in your office?

Michael: Ummm-hmmm...

Stanley: I thought that was the incentive prize for the top salesperson.

Michael: Very easy to clean.

[Stanley walks out]

Michael: Okay. Like he's going to win anyway, right? [laughs]
Michael: Did we get any mail?
Pam: Yeah, I gave it to you.

Michael: Yes you did. Yes, you did. Just checkin'. Just checkin', double checkin', checkin' on the check. Thoroughness is very important in an office and...

Pam: So, can I..? [points to the door]

Michael: Yeah, yeah, of course. Uh, Pam, one more thing. Um, how do girls your age feel about futons?
Jim: A futon?
Pam: [nods]

Jim: He's a grown man

Pam: That's what he said.

Jim: That's sad. Or it's innovative. Well, you know the futon is a bed and couch all rolled into one. [Jim sees Roy and trails off]

Roy: What's up?

Pam: [not looking at Roy] Hi.

Roy: Are you still mad at me?

Pam: Roy...

Roy: Come on [begins to tickle Pam]

Pam: Cut it out.

Roy: Come on, you mad at me?

Pam: Stop it. [laughing]

Roy: Are you still mad at me now?

Pam: [giggling] Cut it out.

Roy: Are you mad at me now?

Pam: Stop. [giggling]

Roy: Huh? huh? Come on... Come on, Pammy I was just kidding.

Pam: [breathless] Stop, I can't breathe.

Roy: I was just kidding. You know I didn't mean it. I can't...
Pam: Jim is a great guy. He's like a brother to me. We're like best friends in the office and I really hope he finds someone.
Katy: You seem to like to touch things. Did you try the velvet?
Angela: I don't like to necessarily touch things. I'm just... I'm shopping.

Katy: Oh no, it's fine that you, um. Here, what about the raspberry one? It's really uh, kind of festive. It's got a lot of personality.

Angela: Yeah, uh no.

Dwight: Hey, how's it going? Good. Can I talk to you for a second? In private?

Katy: I don't think so I'm really busy.

Dwight: It will just take a second.

Katy: I can't.

Dwight: Just for a minute.

Katy: I really can't.

Dwight: Please? I wanted to talk to you in private because I wanted to ask you out on a date.

Katy: No.

Dwight: Ok was that no to talking to me in private, or was that no to the date?

Katy: Both.

[Dejected, Dwight walks out slowly]

Katy: What colors do you like?

Angela: Gray. Dark Gray. Charcoal.
Michael: Ryan.
Ryan: Yeah.

Michael: Would you like to help me with a special project?

Ryan: I would love to.

Michael: Alright.
Michael: [in Michael's car] Okay, just throw out all the empties.
Ryan: You don't want to recycle them?

Michael: Um, yes. Throw them away in the recycling bin.

Ryan: Do you want this? [holding a full bottle of water]

Michael: No.

Ryan: What about this bottle of power drink?

Michael: Uh, what flavor?

Ryan: Blue.

Michael: Blue's not a flavor.

Ryan: It says flavor: Blue Blast.

Michael: Oh, Blue Blast. Yes, put that in the trunk, and there should be an unopened Arctic Chill back there. I want that in the passengers cupholder. Thank you.
Jim: Hi.
Katy: Hi.

Jim: I'm Jim, by the way.

Katy: I'm Katy.

Jim: Hi Katy, nice to meet you.

Katy: You sit out there, don't you?

Jim: I do. That's what I'm best known for. Sitting out there. Alright, let's talk about purses.

Katy: Okay, um...

Jim: Katy but you know what, don't try to sell me one. Okay, seriously 'cause I'm just here to learn.

Katy: Okay. [laughs]

Jim: Okay, so I know about most of these, but you know you can...

Katy: Okay.
Michael: What, stop! Whoa! That's my Drakkar Noir.
Ryan: No, this is Rite Aid Night Swept.

Michael: No, it is a perfect smell-alike. I'm not paying for the label. Right here. Give it.

Ryan: Well, it's empty.

Michael: Not it's not, there's some in the straw. [Michael opens bottle and wipes straw along his neck] There, now you may throw it out.

Ryan: Wow. How many filet-o-fishes did you eat?

Michael: That's over several months, Ryan.

Ryan: [Under his breath] Still.
Jim: What's up?
Pam: I'm bored.

Jim: Thank you for choosing me.

Pam: No, I'm kidding. Um, so you got big plans this weekend?

Jim: Ah, well I think I'm gonna see Katy.

Pam: Really?

Jim: Yeah.

Pam: What are you guys going to do?

Jim: Oh, man I don't know. Uh, dinner, drinks, movie, matching tattoos.

Pam: That's great.

Jim: And stuff... yeah.

Pam: That's cool.

Jim: What are you doing?

Pam: I, I was gonna say, I think that um, we're gonna help Roy's cousin move.

Jim: Okay.

Pam: 'Cause Roy's got a truck.

Jim: That's cool.

Pam: Uh, huh. Yes.

Jim: That is cool. Well, I'll see you Monday though, right?

Pam: Great.

Jim: Okay.

Pam: Okay, I'm gonna head back.

Jim: Alright.
Michael: I think in order to be a ladies man, it's imperative that people don't know you're a ladies man, so I kind of play that close to the chest. I don't know, what can I say? Women are attracted to power. And I think other people have told me that I have a very symmetrical face. [laughs] I don't know. I don't know. Maybe they're right? I don't know.
Michael: Sure you don't want me to help you with that? Cause I can grab that no problem.
Katy: Goodnight, it was nice nice to meet some of you.

Michael: See you later. Goodnight. Goodnight, Jim.

Jim: Goodnight, Michael.

Michael: Where you going?

Jim: I don't know. Grab a drink, I think?

Michael: With us?

Katy: I uh, I probably should have told you, I don't need a ride now 'cause Jim can take me home after so you're off the hook.

Michael: Okay. Great. Off the hook. Excellent. Okay, cool.

Jim: I got this. [taking Katy's bag from Michael]

Michael: Alright, have fun.

Katy: Thanks.

Jim: I got it.

Michael: Don't drink and drive.

Michael: Take it easy.

Jim: Have a good night.

Michael: You too, have a good night.

Katy: You got that?

Jim: Oh, yeah. You sold a lot, so it's lighter.

Katy: Good. Here. Squeeze it inside.

Jim: Alright now, I'm gonna warn you. Don't freak out, okay?

Katy: Why?

Jim: This is a really nice car. In case you haven't noticed, this is a Corolla. Okay.

Katy: It's a... it's a very nice car.

Jim: You're not going to freak out?
Michael: Do I have a special someone? Uh well, yeah of course. A bunch of 'em. My employees. If I had to choose between a one-night-stand with some stupid cow I pick-up in a bar, and these people? I'd pick them every time. Because with them, it is an everyday stand and I still know their names in the morning.
Michael: Tonight is the Dundies, the annual employee awards night here at Dunder Mifflin. [holds up a trophy of a business man] And this is everybody's favorite day. Everybody looks forward to it, because, you know, a lot of the people here don't get trophies, very often. Like Meredith or Kevin, I mean, who's gonna give Kevin an award? Dunkin' Donuts? Plus, bonus, it's really, really funny. So I, you know, an employee will go home, and he'll tell his neighbor, "Hey, did you get an award?" And the neighbor will say, "No man. I mean, I slave all day and nobody notices me." Next thing you know, employee smells something terrible coming from neighbor's house. Neighbor's hanged himself due to lack of recognition. So...
Jim: So, you ready for the... the Dundies?
Pam: Ugh...
Pam: You know what they say about a car wreck, where it's so awful you can't look away? The Dundies are like a car wreck that you want to look away, but you have to stare at it because your boss is making you.
Michael: [in a Fat Albert voice] Hey hey hey! It's Fat Halpert.
Jim: What?

Michael: [in Fat Albert voice] Fat Halpert. [in normal voice] Jim Halpert.
Michael: So why don't I take you on a tour of past Dundie winners. We got Fat Jim Halpert here. Jim, why don't you show of your Dundies to the camera?
Jim: Oh, I can't because I keep them hidden. I don't want to look at them and get cocky.

Michael: Oh, that's a good idea.

Dwight: Mine are at home in a display case above my bed.

Michael: Gyaaah. T.M.I. T.M.I my friends.
Michael: T.M.I.? Too much information. Ah, it's just easier to say T.M.I. I used to say "Don't go there" but that's... lame.
Michael: And here we have Stanley the Manly. Now Stanley is a Dundie all-star, aren't you Stan? Why don't you, ah, show them some of your bling.
Stanley: I don't know where they are, I think I threw them out.

Michael: Oh, no you di-int.

Stanley: I think I did.

Michael: W-why did you...

Stanley: Say, we got to order some more apa-teezers this time. We ran out last year, remember?

Michael: Yes we should. I... you know what? I wanted one of those skillets of cheese, but when I got off stage, [turns to Kevin] someone had eaten all of them.
Michael: [in video] To Oscar Martinez it's the "Show Me the Money" award! Yeah!
Pam: Michael has taped every Dundies awards and now, he's making me look through hours of footage to find highlights.
Oscar: [in video] That's supposed to be confidential.
Michael: [in video] He has the award-ah! ...it's a type of song that we are going to play for the ladies. Hit it, Dwight!

[Dwight starts playing the tune of "Mambo No. 5" by Lou Bega on his recorder]

Michael: [singing along to tune on video] A little bit of Paaam, all night long, a little bit of Angela on the thing...

[Somebody sits in front of the camera on the video, so even though nothing can be seen, Michael can still be heard]

Michael: [in video]...a little bit of Phyllis everywhere...

Pam: Oh, yeah, this is the part where Kevin sat in front of the camcorder all night. It's great.

Michael: [on video] ...a little bit of Roooy eating chicken crispers... ...a little bit of Jim with some ribs, a little bit of...
Kelly: It was you.
Phyllis: Live and learn.

Pam: [quietly laughing] It wasn't. I swear.

Kelly: Yeah, it was.
Dwight: So, what's the joke? You're not perfect either.
Pam: We're not laughing at you, Dwight.

Dwight: So who are we laughing at?

Pam: Um, just something somebody wrote.

Dwight: Who? Dave Barry?

Kelly: [laughing] No. No, just something that was written in the ladies' room wall.

Dwight: What is it? Who wrote it?

Pam: Um, it's kind of private.

Phyllis: [whispering] It's about Michael.

Dwight: That is defacement of company property. So you better tell me. Kelly, if you tell me, you'll be punished less.

Pam: Okay, now I'm laughing at you.
Michael: [talking to the speakerphone] Will her highness, Jan Levinson-Gould, be descending from her corporate throne this evening to visit us lowly serfs here at Dunder Mifflin Scranton?
Jan: [on speaker phone] It's a, it's, it's a two and a half hour drive from New York, Michael.

Michael: Well, you could take the bus. You could work on the way here. Sleep on the way home.

Jan: No.

Michael: Wuh... Come on, Jan. This is important. I mean, this is, this is, validation to my employees here that you and corporate approve of this. So...

Jan: No, we don't approve of this Michael. I mean, y-you only had the budget for one office party a year, so... we're not paying for this.

Michael: Um...

[Michael looks at the camera and motions for the camera to leave the office]

Michael: [to camera] Could you...?

Jan: Are you there Michael?

Michael: Yeah, I'm here, I just wanted to, uh, talk to you for a second about that.

[Michael closes the blinds]

[The camera tries to find a crack in the blinds]

Michael: Um, what, ah, what is, I mean...

[The camera pans around to reception, Pam is listening]

Michael: ...come on, Jan!

[The camera goes to a side of Michael's office where the blinds are still partially open]

Michael: You're dropping an A-b*mb on me here.

Jan: Really? I'm dropping an atomic b*mb on you?

Michael: Well, yeah, I mean, what is...

Jan: You already had a party on May 5th for no reason.

Michael: No reason?! It was the 05 05 05 party...

Jan: And you had a luau....

Michael: ...it happens once every billion years.

Jan: And a tsunami relief fundraiser which somehow lost a lot of money.

Michael: Okay, no, that was a FUN raiser. I think I made that very clear in the fliers, fun, F-U-N.

Jan: Okay, well, I don't understand why anyone would have a tsunami FUN raiser, Michael. I mean, that doesn't even make sense.

Michael: Well, I think a lot of people were very affected by the footage.
Michael: This is a little character I like to do [places a green turban with a yellow feather on his head], it is, uh, loosely based on Karnack, one of Carson's classic characters. [puts an envelope to his head] Here we go. The PLO, the IRA, and the hot dog stand behind the warehouse. [tears open envelope and pulls out card] "Name three businesses that have a better health care plans than Dunder Mifflin." Here's the problem. There's no open bar because of Jan and it's the reason why comedy clubs have a two drink minimum. It'll be fine, I just...wish people were going to be drunk.
[Phyllis catches Dwight trying to sneak into the girls bathroom]
Phyllis: Dwight, get out of here!!

[The door swings open and Dwight is being pushed out by Phyllis]

Dwight: No, no, no, no...

Phyllis: What were you doing in the ladies room?!

Dwight: ...no, no, no, no, it's not what you think.

Phyllis: Why were you peering over the stalls?!

Dwight: No, why were you in there?!

Phyllis: You are a pervert!

Dwight: What were you doing in there?

Phyllis: You, are, a pervert!

Dwight: I am not.
Michael: [in video] The Dundie award for "Longest Engagement" goes to Pam Beesley.
Michael: Pam, everybody! [starts clapping]

[Pam just sits there stirring her drink, rolls her eyes and glances over at Jim]

[Jim, at the adjacent table, crosses his arms and glances over at Pam, both look annoyed]

Michael: Whoooo! When is that girl gonna get married? That's what I have to say. Ah, Roy's accepting.

Roy: [on video] Yes.

Michael: [on video] Thank you Roy. Are there any words you'd like to say, on Pam's behalf?

Roy: [on video] Ah, w-we'll see you next year.

Michael: [on video] Yeah, oh, hope not! Oh God!
Michael: I'm not changing that, it's the best one.
Jim: No, it's hilarious, you're right. I just think, um, "world's longest engagement", um, we're all expecting it, you know?

Michael: That's why it's funny. Every year that Roy and Pam don't get married, it gets funnier.

Jim: Well I think if you use the same jokes it just comes across as lazy.

Michael: Oh, [taking it to heart] lazy. Uh huh.
Dwight: Excuse me, everyone, could I have your attention please. I just wanted to say that the women in this office are terrible. Especially the ones who wrote that stuff about Michael on the bathroom wall. Having a bathroom is a privilege. It is called a ladies room for a reason. And if you cannot behave like ladies, well then you are not going to have a bathroom.
Pam: You're taking away our bathroom?
Dwight: We are going to have two men's rooms.

Phyllis: But where would we...go?

Dwight: Be prepared to hold folks [Michael comes out of his office] From 9 am to...

Pam: Michael...

Michael: Yes.

Pam: ...Dwight is banning us from our bathroom.

Michael: Okay, well, that's just ridiculous, so just don't, I-I don't have time for this right now.

Dwight: Nnnnno, there needs to be repercussions...

Michael: Just don't, don't talk-

Dwight: ...for people's behavior.

Michael: Don't talk-

Dwight: And it's-

Michael: Just STAP IT YAP IT!!!!
Michael: Okay, look, I know there have been a lot of rumors flying around about the Dundies this year. How there is no money, and how there is no food, and how the jokes are really bad, but WHAT THE HELL EVERYBODY!? I mean, God. The Dundies are about the best, in every, one of us. Can't you see that? I mean, okay, we can do better. so, tonight, for the first time, we are inviting all of your friends and family to attend the awards with us.
Dwight: [with a small fist pump] Yes!

Michael: Yeah, not bad, right? So let's make this the best Dundies ever.

Dwight: [clapping] Best Dundies ever.
Dwight: Welcome to the eighth annual Dundies awards.
[Quick cut to everybody talking and ignoring Dwight]

Dwight: Before we get started, a few announcements. Keep your acceptance speeches short, I have wrap it up music, and I'm not afraid to use it. [points] Devon!
Michael: "The Dundies, how can I explain it? Awards you like to hate it. I'm psyched you all made it. You never had to work so hard and feel that no one notices you. You're just a name and number and no one even says hello." [to Ryan] Card!
Oscar: The Dundies are kind of like a kid's birthday party, and you go, and there's really nothing for you to do there. But the kid's having a really good time, so you're, kind of there. That's-that's kind of what it's like.
Michael: "You down with The Dundies? You down with the Dundie-"
[The music stops, Michael looks back at Dwight]

Dwight: The waitress tripped on the cord.

Michael: Alright, alright, joke landed. So we are here, thank you all for coming to the 2005 Dundie awards. [takes off sweater to reveal tuxedo] I am your host, Michael Scott. And I just want to tell you please, please, do not drink and drive. Because you may hit a bump and spill the drink!
Kevin: [to waitress] Oh, just put these on the group tab.
Michael: Nope, actually this year, ah, no group tab, we're going to be doing separate checks.

Stanley: You said, we could bring our families.

Michael: I did. And why didn't ya Stanley?

Stanley: I did, my wife's name is Terri.

Michael: Well, I'm looking forward to meeting Terri.

Stanley: It's this person who's hand I'm holding Michael.

[Michael is dumbfounded, Dwight pushes a button on his keyboard that says, "OHHH, YEAHHHH."]
Michael: [to Dwight, in a low voice] Shut it. [normal voice] Um, good. Speaking of relationships, of all, all way shapes and forms. Um, I was out on a very, very hot date with a girl from HR, Dwight.
Dwight: Really? We don't have any girls from HR.

Michael: No, that...for the sake of the story. And things were getting hot and heavy.

Dwight: Yeah?

Michael: And I was about to take her bra off...

Dwight: Yeah!

Michael: ...when she made me fill out six hours of paperwork-

Dwight: Like an AIDS test?

Michael: No! [under his breath] God.
Michael: [clears throat] Alright, so let's get this party staaaarrrrted.
Darryl: Hey let's go to Poor Richard's.
Roy: Yeah, let's get out of here.

Pam: Um...
Michael: Um, guys, where you going? Pam, show's just getting started.
Pam: Sorry.
Ryan: You staying?
Jim: Yeah, gotta eat somewhere.
Michael: And now... to someone who quietly goes about their job, but always seems to land the biggest accounts...
Michael: ...the "Busiest Beaver" award goes to Phyllis Lapin.

[Everybody starts clapping, Phyllis gets out of her booth and makes her way to Michael, she gives Jim a high five along the way]

Michael: Yeah, way to go Phyllis. Nice work, per usual.

Phyllis: This says "Bushiest Beaver".

Michael: What? I told them busiest...idiots.

Phyllis: It's, it's fine.

Michael: Well, we'll fix it up. You don't have to display that.
[Pam and Roy are at the truck, arguing.]
Pam: ...because that's what happens every time!

Roy: ...talking about? He's a jackass every year.

Pam: No.

Roy: [Put's his hand on Pam's arm] Come on, we're going to Poor Richard's.

Pam: [Breaks Roy's grip] No, I don't want to go, I don't want to.

Roy: Pam. Go.

Pam: If you would have asked me that, then you would know.
[Michael has false teeth in and glasses with squinted eyes on them]
Michael: [in a stereotypical oriental accent] Herro everybodeeee. I'm gonna cwall Jan Revinson-Gould.
Jim: Hey! How are ya? I thought you left?
Pam: Oh no, I just, I decided to stay.

Jim: Oh!

Pam: I'll just get a ride home from Angela.

Jim: Oh.
Pam: Oh good, I'm just in time for Ping.
Jim: Yeah.

Michael: [doing impression] Oh, dat wir be fwar. Ah, me so horny.

[The camera zooms to an Asian customer behind Michael, she is looking at Michael in disbelief]

Michael: Right? You know wat I'm talking 'bout.
Pam: [to an off camera waiter and still clutching Jim's beer] Can I get a drink?
Michael: This next award goes to somebody, who really, lights up the office.
[Cut to Pam still drinking the beer]

Michael: Somebody, who I think a lot of us, cannot keep, from checking out. The "Hottest in the Office" award goes to... ...Ryan the temp!

Michael: Yeah. [singing to music] "Hidy ho, you sexy thang. You sexy thang you." Here you go.
Ryan: What am I going to do with the award? Nothing. I-I don't know what I'm going to do. That's the least of my...concerns right now.
Michael: And the "Tight Ass" award goes to Angela. Not only because she is everybody's favorite stickler, but because she has, a great caboose. So...come on down.
Angela: No.
Jim: [Pam starts sipping an empty glass] I think those might be empty.
Pam: No, no. 'Cause the ice melts and then it's like second drink! [laughs]

Jim: Second drink?
Michael: The "Spicy Curry" award goes to our very own Kelly Kapoor! Get on up here. Here you go.
Kelly: "Spicy Curry", what's that mean?

Michael: Um, not everything means something, it's just a joke.

Kelly: Yeah, but why'd you give it to me?

Michael: I don't know, it's just...

Kelly: This is a bowler-

Michael: I know. It's ju- they didn't have any more businessmen. So...

Kelly: Yeah, but everyone else-

Michael: Just sit down Kelly.
Michael: [sweaty and chugging water from a bottle] It's so freakin' hot in there. Now I know what Bob Hope was going through when he performed in Saudi Arabia. Man! I got Dwight sucking the funny out of the room, but you do what you can do. [Music starts playing in the background] Here we go, he's early with the cue. Here we go.
Michael: [Michael is singing to the tune of "Tiny Dancer" by Elton John] "You have won a tiny Dundie."
Guy at bar: Sing it Elton.

Michael: Hey, thanks guys. Hey, where you guys from?

Other Guy at Bar: We just came from yo' mama's house.
Michael: Oh, alright, yeah.
Guy At Bar: Sing 'em a song dude.

Michael: Uh, you know what guys, we're just having a little office party, so if you want, uh...

Michael: [Something flies by Michael] Hey, you know, cool it guys, really-

[The guy at the bar throws another object, looks like a wad of wet napkins, this time it hits Michael on the shoulder]

Guy At Bar: You suck man!

Michael: Let's cut it. [Dwight turns the music off]

Michael: [clears throat] [with a lot less enthusiasm] I had a few more Dundies to, uh, give out tonight, but, I'm just going to cut it short. And wrap it up so everybody can enjoy their food. Um...thanks for listening, those who listened. [clears throat] This last Dundie is for Kevin, this is the "Don't Go in There After Me" award. It's for the time that I went into the bathroom after him, and it was really, really smelly. So...

Michael: [give Kevin his award]There you go.
Pam: Yay Kevin. Whoo hoo for Kevin! For stinking up the bathroom.
Jim: [starts clapping] Yeah, alright Kev.

[More people start clapping]

Pam: Woo! Hey, I haven't gotten one yet!

Jim: Yes, I have not gotten one either. So, keep going.

Pam: More Dundies!

Pam and Jim: [clapping] Dundies! Dundies! Dundies! Dundies!

Everybody: Dundies! Dundies!

Michael: [getting his spirit back] Alright, alright, alright, okay. Alright, we'll keep rolling. Okay, this is the fine work award. This goes to Stanley, for all the fine work he did this year.

Pam: Fine work! Fine work Stanley!

Michael: You know you did.
Pam: Here here! Speech, speech, speech, speech [other people start joining in]
Stanley: Well, well, last year, I got great work, so I don't...

[Pam starts laughing her cute drunk laugh]

Stanley: So, I don't know what to think about this award. But at least I didn't get smelliest bowel movement like Kevin. [starts chuckling]
Michael: And this next award is going out to our own little Pam Beesley...
[Cut to Pam, her face goes from drunken elation to sober realization]

Michael: ...I think we all know what award Pam is going to be getting this year.

[Cut to Jim's reaction of scared expectation]

Michael: It is the "Whitest Sneakers" award! Because she always has the whitest tennis shoes on!
Michael: Get on down here! Pam Beesley ladies and gentlemen! [Pam grabs the microphone from him] Oh, here we go.
Pam: I have so many people to thank for this award.

[Quick cut to Jim laughing and staring at Pam with amused wonder]

Pam: Okay, first off, my Keds. Because I couldn't have done it without them. [people clap] Thank you. Let's give Michael a round of applause for MC-ing tonight because [people start clapping again] this is a lot harder than it looks. And also because of Dwight too.

[Dwight stands up, but nobody claps]

Pam: Um, so, finally, I want to thank God. Because God gave me this Dundie.

[Quick cut to Jim, he's doesn't know whether to laugh or take her seriously, so he gives her an amused/appreciative grin]

Pam: And, I feel God in this Chili's tonight. WHOOOOOOOO!!!!

Michael: Pam Beesley ladies and gentleman. [Pam hugs Michael and gives him a quick peck on the cheek] Oh! Thank you.
Jim: What a great year for the Dundies.
Jim: We got to see Ping. [Pam nods] And we learned of Michael's true feelings for Ryan. [Pam nods] Which was touching. And, we heard Michael change the lyrics to a number of classic songs. [Pam nods] Which for me, has ruined them for life. [looks at Pam, who is staring at him, nodding]
[Jim looks at the camera, then back at Pam, who is still nodding]

Jim: What?

Pam: Nothing.

Jim: Okay.

Pam: What?

Jim: I don't know, what?

[Pam starts laughing, then suddenly falls off the bar stool]

Jim: Oh my God! You are so drunk!
Jim: Did you get that? Please tell me you got that. This is all going to be on.
Dwight: Quick, quick, the woman is having a seizure. Grab her tongue, grab her tongue! It's okay, I'm a sheriff's deputy.

Jim: He's a volunteer.

Dwight: Don't get into that now. We need something to cushion her head. Throw pillow? A cush-

Jim: Dwight come on, come-

Dwight: It's okay, I'm going to use my shirt.

[Dwight starts taking off his shirt, but gets stuck]

Pam: Dwight, get off me!

[A Chili's employee comes over, Jim helps Pam up, Dwight is stuck in his shirt]

Employee: I'm sorry, you're gonna have to put your clothes back on, people are trying to eat.

Dwight: [struggling] Ahh! I can't-
Michael: Was this year's Dundies a success? Well, let's see, I made Pam laugh so hard, that she fell out of her chair, and she almost broke her neck. So I k*ll, almost.
Pam: Oh my God!
Jim: Whoa.

Pam: I just want to say, that this was the best, Dundies, ever! WHOOOOOOOOO!!!

Jim: Whoa.

Jim: Whoa, careful, careful.
Chili's Employee: We have a strict policy here not to over serve. Apparently, this young woman was sneaking drinks off other people's tables. I Xeroxed her driver's license and she is not welcome to this restaurant chain ever again.
Michael: Great work tonight.
Dwight: Watch your step.

Michael: Excellent.

Dwight: Thanks, I had to, uh, check her pupils to make sure there wasn't a concussion.

Michael: Yes, that too, but I mean with the audio. Great work.
Pam: I feel bad about what I wrote on the bathroom wall.
Jim: No you don't.
Jim: Oh, here she is. Careful, careful, whoa. Alright, easy. Almost there.
Pam: Hey, um, can I ask you a question?

Jim: sh**t.

[Pam stares at Jim for a little while, then glances at the camera, realizes she's on camera]

Pam: Um, I just wanted to say thanks.

Jim: Not really a question. [starts to laugh] Okay, let's get you home, you're drunk.

[Jim opens the door for her]

Jim: Alright.

Pam: Bye.

Jim: Goodnight, have a good night. Thank you Angela.
Michael: [clears throat] Hey, what's up?
Jim: Hey.

Michael: Any emails today?

Jim: Um... I don't think so.

Michael: No? Um... Check your spam folder.

Jim: Oh! There it is!

Michael: What?

Jim: Um... 'Fifty signs your priest might be Michael Jackson.'

Michael: [laughs uncontrollably]

Jim: Well done.

Michael: Kay.

Jim: Topical.
Michael: I am king of forwards. It's how I like to do business, everybody joking around. We're like 'Friends'. I am Chandler and Joey and, uh, Pam is Rachel. And Dwight is Kramer.
Dwight: So the monkey does the sex thing right here! [monkey noises in background]
Michael: That's funny! That's funny. Not offensive. Uh... because it's nature. Educational.

Dwight: Do you want the link because then you could forward it around?

Michael: Um, I...

Dwight: Consider it?

Michael: Yeah... maybe. Maybe. Well, we'll see. Because I... I don't know if it's... [muffled by jacket over his head] Whup! Come on! Hey!

Todd Packer: What has two thumbs and likes to bone your Mom? [points at self] This guy!

Michael: Kay! Oh, you are so bad! Yeah!

Todd Packer: [makes laser g*n noises]

Michael: Oh, Boom! Bam! Oh, this guy is out of control! He is a madman! Better get the bleep button ready for him.

Todd Packer: bleep, bleep. What's up, Halpert?

Michael: Uh oh.

Todd Packer: Still q*eer?

Michael: Uh oh! Oh-ho-ho-ho-ho-o!
Michael: Todd Packer and I are total BFF. Best Friends Forever. He and I came up together as salesmen. One time, we were out and we met this set of twins. And Packer told them that we were brothers. And so, you know, one thing led to another, and we brought em back to the motel. And then Packer did both of them. It was awesome. So...
Michael: Oh-whoa-oh! Oh! Okay. Grade 'A' gossip for you, right now. Randall, CFO, resigned. Nobody knows why.
Todd Packer: Are you kidding? Everyone knows why! You don't know? Okay, check this out. Al lright. So here's the story. So Randall is nailing his secretary, right? And she is totally incompetent.

Michael: Really? Here we go! Buckle up. It's going to be a bumpy one!

Todd Packer: We're talking blonde incompetent.

Michael: Oh, yeah.

Todd Packer: Like 10 words a minute... talking.

Michael: Well, to be fair... blondes, brunettes, you know, there's a lot of dumb people out there.

Todd Packer: They are women, right?

Michael: Oh! Wow! I didn't say it! I didn't say it!

Todd Packer: I said it. And then, suddenly, for no reason, this bimbo blows the whistle on the whole thing just to be a bitch.

Michael: Oh, wow! What did I tell you about the bleep button.
Jim: Hey, um... what has two thumbs and hates Todd Packer? [points at self] This guy!
Todd Packer: Meant to ask you, can you think you can get someone to drive me around because of the, uh, DUI situation?
Michael: Oh. Bad boy. [to Ryan] Um... Ryan? [makes Donald Duck noise]

Todd Packer: [to Ryan] Come on, kid. Let's go.

Michael: Ah! Man. That Todd Packer can do anything.

Jim: Except pass that breathalyzer.
Ryan: You a big William Hung fan?
Todd Packer: Why does everyone ask me that? Who the hell is that?
Kevin's computer: [monkey noises]
Jim: I'm really excited to meet your Mom.
Pam: You are?
Pam: My Mom is coming in to visit. And she lives like two hours away. And she doesn't have a cell phone... which is cool cause it's kind of adding some suspense to my day. And I keep looking over at the door hoping she'll walk in.
Pam: I've decided to show her around. She really wants to meet everybody.
Jim: Oh yeah?

Pam: mmhmm.

Jim: Good. Cause I have a lot of questions.

Pam: Oh really?

Jim: Yeah. As a child, did Pam show any traits that would hint towards her future career as a receptionist?
Michael: Hey, send me that link to the monkey sex video. I'm going to forward it like it's hot.
Dwight: Yes!

Michael: Forward it like it's hot. Forward it like it's hot. "Old School".

Toby: Michael?

Michael: Yes, Toby?

Toby: Um... I need to talk to you in your office. It'll just take two seconds.

Michael: Um... literally two seconds?
Michael: Toby is in HR which technically means he works for Corporate. So he's really not a part of our family. Also he's divorced so he's really not a part of his family.
Toby: The full story is that Randall resigned because of sexual harassment. So Corporate asked me to do a five minute review of the Company Sexual Harassment policy.
Michael: No, no, Toby. No.

Toby: It's really not a big deal, Michael.

Michael: It is a big deal. It's a big deal! What are we supposed to do? Scrutinize every little thing we say and do all day? I mean, come on!

Toby: And then Corporate is going to send in a lawyer...

Michael: What?

Toby: Just to refresh you... .

Michael: NO!

Toby: on our policy.

Michael: What? He! No! Okay, what is a lawyer going to come in and tell us? To not send out hilarious emails or not tell jokes?

Toby: Maybe not some of them. Maybe not inappropriate ones.

Michael: There is no such thing as an appropriate joke. That's why it's a joke.
Michael: Everyone! Hello! Everyone. Hi! Sorry to interrupt. I know you're all busy and the last thing you want is for a major interruption. But Toby has an announcement that he insists on making right now in the middle of the day. [to Toby] So, take it away.
Toby: Yeah, okay. Corporate would like us to do a five minute review of the Company Sexual Harassment policy so I'll go over that later.

Michael: I wish you luck, Toby. I really do. But you are going to have a mutiny on your hands and I just can't wait to see how you handle it.
Michael: A guy goes to a five dollar... lady of the night and he gets crabs. So, the next day he goes back to complain. And the woman says "Hey. It was only five dollars. What did you expect? Lobster?" This is what's at stake.
Michael: Time to bring out the big g*n. I'm heading down to the warehouse where jokes are born. Find a k*ll joke that'll just blow everybody away at the seminar later. And remind them what is great about this place. So... ah! Here they are. [to Warehouse guys] Guys! Wondering if I could, uh, get your help for something. I'm looking for a new joke to tell and it needs to be just k*ll. And it does not need to be clean. So whatcha got?
Darryl: Like a joke? A knock-knock joke?

Michael: Um, yeah, no, well... I mean better. Better than that. The type of stuff you guys tell all day.

Darryl: Well, [points at Michael] those are some awful tight pants you have on. Where'd you get em? Like Queers R Us?

Roy: Boys R Us!

Warehouse Guy: Oh!

Michael: Alright, alright. Well, yeah, but, you know... a joke but not necessarily at my expense.

Darryl: Man, we can see all your business coming around the corner, okay? You need to, you know, hide the... good thing you don't have a lot of business to start with.

Michael: Oooh, okay. That was still about me.

Roy: Hey, hey, hey.

Michael: What?

Roy: So you don't have the biggest package. Don't feel bad.

Michael: I don't feel bad.

Darryl: [fake whispers to Roy] I think he feels bad.

Michael: No, I don't.

Roy: You look like you feel bad.

Michael: Okay.

Roy: Little package!

Michael: Well, not exactly what I was looking for but thanks guys.

Warehouse guy: Little package! Little package!

Michael: Thank you.

Roy: You look good.

Darryl: Hiding from his momma.

Warehouse guys: [kissing noises, sheep baaing sounds]
Toby: So remember, intent is irrelevant. And that's it. Pam?
Pam: Um... I just wanted to say that... Just, my Mom's coming in today.

Kevin: MILF!

Pam: Thanks, Kevin.
Pam: Usually the day we talk about sexual harassment is the day that everyone harasses me as a joke.
Pam: She's coming in today and maybe just don't joke around about that stuff in front of her.
Toby: Great point.

Pam: Thank you.

Toby: Um... in fact, basic rule of thumb, let's just act everyday like Pam's Mom's coming in. All right. That's it. Um... if anybody has any questions about anything, you know where I sit in the back.

Michael: Hi, is it over?

Toby: Uh, yes!

Michael: No.

Toby: I can go over it with you.

Michael: I know, I know. It's good. It is not over. It is not over til it's over.

Toby: It's over.

Michael: Did he tell you everything? Obviously, he didn't because you all still look relatively happy. Albeit bored. Do you realize what we're losing? Seriously?

Angela: Email forwards.

Michael: Exactly! Mmwwah [blows kiss to Angela]! Can we afford to lose email forwards? Do we want that?

Angela: I hate them. You send me these filthy emails and you say forward them to ten people or you'll have bad luck.

Michael: Give me a break. Umm... Stanley, how about that hot picture you have by your desk? Centerfold in the Catholic schoolgirl's outfit? I mean, it is hot, it is sexy, and it turns him on. And I will admit, best part of my morning is staring at it. But what? Are we just going to take it away?

Stanley: That is my daughter. She goes to Catholic girls' school. I am taking it down right now.

Meredith: Um... what about office romance?

Toby: Office relationships are never a good idea. Yeah. So let's just try to avoid them. But, um, if you already have one, you should disclose it to HR.

Phyllis: All relationships? Eh, even a one-night stand?

Michael: I think the old honor system was just fine. For example, I have never slept with an employee. And, believe me, I could have.

Dwight: Yeah, Meredith.

Michael: No! No! Catherine. Remember her? Remember how hot she was?

Dwight: Yes.

Michael: She would have definitely slept with me.

Kevin: She wasn't that hot.

Michael: Yes, she was. Dammit, Kevin!

Toby: Ok, you know, Michael...
Jim: I'm in an office relationship. It's special. Um... she's nice. She's shy. She's actually here. You want to meet her? Hold on one second. Oh, my God! Put on a shirt! Put on a... . I told you that you'd be on camera. I'm sorry, she's European. No, I told you that you'd be on camera. Stop it.
Michael: What if Pam was a lesbian? What if she brought her "partner" in to work? [to Toby] Would that be crossing the line?
Toby: No.

Michael: What if they made out? In front of everybody?

Toby: Well, that would be...

Michael: At home? And I told everybody everything about it.

Toby: Okay, I'm lost.

Michael: Okay. Well, then let's act it out. Pam, you will be girl A and girl B will be... Okay! We'll use the doll. Pam. Pam?
Video: [Crossing the Line: Rules for the Modern Workplace]
Michael: I wish Todd Packer was here because he would love this. I wonder if anybody else would like to do this. Hey! Um... we have to watch, uh, Toby's video that he's showing us in order to brainwash us and I was wondering if anybody would like to join in? Going to be fun. Got my great pizza. Whataya say? Jim?

Jim: No, thanks. I'm good.

Michael: That's what she said. Pam?

Pam: Uh... my mother's coming.

Michael: That's what she sai [clears throat] Nope, but... Okay. Well, suit yourself.
Dwight: Hey, Toby.
Toby: Hey Dwight.

Dwight: You said that we could come to you if we had any questions.

Toby: Sure.

Dwight: Where is the clitoris? On a website, it said at the crest of the labia. What does that mean? What does the female vagina look like?
Toby: Technically, I am in Human Resources. And Dwight was asking about human anatomy. Um... I'm just sad the public school system failed him so badly.
Toby: Yeah, maybe when you get really comfortable with each other, you can ask for that.
Dwight: Good. Good. And...

Toby: I should get back to work.

Dwight: Okay.
Man in Video: In today's fast-paced business climate, it can sometimes be hard to know when a comment or an action crosses the line. Let's take a look at a couple of scenarios and ask ourselves 'where is the line?'
Video: [Scenario 1[/b]: The Natural Redhead]

Roy: Natural redhead.

Actor: Hey, Rach.

Redheaded Actress: Hey, Joe. Mike.

Actor: Hey, settle a bet. Are you a natural redhead?

Darryl: Oh, Mi... ! Hey, stop the video! Michael, stop it right there! Stop it right there! That's that girl from that thing. [pointing at Redheaded Actress] I banged this girl right here. This is...

Roy: That's her?

Darryl: Yes, this is the one.

Roy: No!

Darryl: You remember? Yes!

Roy: At the party?

Warehouse guy: You banged her?

Darryl: Yes! [to video screen] Right here. You are a naughty girl!

Michael: Whoa, whoa, whoa, whoa... Okay! Hypocrite! She is a hypocrite. That is such a scam! Okay.
Jan: [to cell phone] Yes. Yes, I did. Okay. Well, we can talk about that later then. [to Pam] Hi.
Michael: Okay, you are never going to believe this. The girl in the video we're watching that Corporate gave us... Darryl banged her! Aaand is about 90% sure.
Todd Packer: Don't ever let this little bitch drive you around town. We got, uh, lost for half an hour.
Pam: I don't have any DUI's so I can drive myself, but thanks.

Todd Packer: Where is Michael Snot? Sniffing some dude's thong? Probably.
Michael: So you are the lawyer, Mr. O'Malley? I know a lot of lawyer jokes.
Mr. O'Malley: I love lawyer jokes.

Michael: Well, it's probably because you don't get 'em.
Michael: When I said before that I was king of forwards, you got to understand that I don't come up with this stuff. I just forward it along. You wouldn't arrest a guy who's just delivering drugs from one guy to another.
Jan: You seem a little bit agitated, Michael. What's the problem?
Michael: The problem is that I am the boss and apparently I can't say anything.

Jan: Well, that... that's true in a way. You can't say anything.

Michael: Where's the line? Where's the line, Jan.

Jan: Do you need to see the video again, Michael?

Michael: No, I've seen the video.

Toby: [to Jan] He talked the whole time.

Michael: No, I didn't. [to Jan] Huh, what? [everyone looks up at blow-up doll]
Michael: Attention, everyone! Hello! Ah, yes! I just want you to know that, uh, this is not my decision, but from here on out... we can no longer be friends. And when we talk about things here we must only discuss work-associated things. And, uh, you can consider this my retirement from comedy. And in the future, if I want to say something funny or witty or do an impression, I will no longer, ever, do any of those things.
Jim: Does that include 'That's What She Said'?

Michael: Mmmhmm. Yes.

Jim: Wow! That is really hard. You really think you can go all day long? Well, you always left me satisfied and smiling, so...

Michael: THAT'S WHAT SHE SAID!

Jan: Michael. MICHAEL!

Michael: [laughing] Come on.

Jan: Michael, please.

Todd Packer: There he is.

Michael: Mwah! [kisses hand and salutes office]

Todd Packer: There he is. Good one.
Michael: You would have done the same. You just didn't think of it first.
Jan: Mike... Michael. Please. I... I... really.

Michael: It's... That's...

Jan: That's not my sense of humor.

Michael: Okay. [to man entering office] Hello. [introduces] Jan. Mr. O'Malley. This is my lawyer, James P. Albiny.

Jan: Wha...

Michael: I believe you may recognize his face from the billboards. He specializes in Free Speech issues.

Albiny: [to camera] And motorcycle head injuries, worker's comp, and diet pill lawsuits.

Michael: This guy does it all.

Jan: [to Albiny] 'Scuse me, I'm sorry. [to Michael] Michael. Mr. O'Malley is your lawyer.

Michael: What?

Jan: Mr. O'Malley is our Corporate lawyer. We have him on retainer. To protect the company as well as upper level management, such as yourself.

Michael: So I'm not in trouble?
Michael: I am so used to being the bad boy. I am so used to fighting Corporate that I forget that I am Corporate. Upper management. They hooked me up with an attorney. To protect me. You can't be too careful about what you say. Mo' money, mo' problems.
Michael: Okay. Well, let's get you out of here, James. Um... I think we're under an hour still, so...
Albiny: Yeah, but I did a lot of paperwork at home before I got here.

Michael: I know. We'll talk about it later. Thanks for coming in.
Pam's Mom: Um... hello.
Pam: [ecstatic] Oh my god!

Pam's Mom: Finally made it!

Pam: Hello!
Pam: I love my Mom. Okay. That's probably really the most obvious statement ever.
Pam's Mom: This is all yours?
Pam: Yeah. I'm in charge of this whole area.

Pam's Mom: Oh, my goodness. That's great.
Todd Packer: So a guy goes home, tells his wife, "Honey. Pack your bags. I just won the lottery." She goes, "Oh my god! That's incredible! Where are we going?" He goes, "I don't know where you're going, just be out of here by five!" [men laugh] Boom!
Pam: This is where I used to keep my computer.
Pam's Mom: Oh, right! I remember...

Pam: But then I moved it.

Pam's Mom: with the picture.

Pam: Yeah, yeah, but I uh... I switched stuff around because I actually needed like more room for organization. So...

Pam's Mom: Sure.

Pam: So this is like, um, an organization station...

Pam's Mom: [to Roy] Oooooh!

Pam: Hey!

Pam's Mom: Well, there he is!

Roy: How are ya?

Pam's Mom: Hi, handsome!

Roy: You look great!

Pam's Mom: Oh, thank you! So! We ready for dinner?

Pam: Well, you know... actually, I kind of need to stall a bit. But, it's okay, because I am very used to k*ll time.

Pam's Mom: Oh, I don't believe that.

Roy: Okay, I'm going to go wait in the parking lot. And what kind of tunes you want for the ride? Little, uh, classical? Or oldies?

Pam's Mom: Oh, anything is fine.

Roy: All right, I'll see ya.

Pam's Mom: So which one is Jim?

Pam: Mom!

Pam's Mom: I just wanted to know.

Pam: No.

Pam's Mom: All right. Okay.

Pam: Ten minutes.

Pam's Mom: Okay.

Pam: Then we can go to dinner.

Pam's Mom: I'll make myself busy.
Todd Packer: There's this guy. He's at a Nymphomaniac Convention. And he is psyched 'cause all these women are smokin' hot perfect 10's, except for this one chick who looks a lot like, uh... [points at Phyllis]
Kevin: Phyllis?

Michael: No. No, no, no. That crosses the line.

Todd Packer: Ex-squeeze me?

Michael: Not you. Kevin. Just unwarranted. Hostile work environment, Kevin.

Kevin: Packer said it.

Michael: No. You said it. He pointed. A point is not a say. Look. Kevin, we are a family here and Phyllis is a valued member of that family. Like a grandmother.

Phyllis: I'm the same age as you, Michael.

Michael: I don't know about that.

Phyllis: We're in the same High School class.

Michael: Well, I have a late birthday and usually September's a cut-off point. [to Kevin] You know what? You just crossed the line. Okay? There's a line and you went over it. And you must be punished. So go to your corner.

Kevin: You mean where my desk is?

Michael: Yes, your corner. Go.

Kevin: Okay. I have a lot of work to do anyway.

Michael: Mmmhmmm.

Todd Packer: Oh my. They really got to you, didn't they?

Michael: They didn't get to me. I got to them. I am still the same old Michael Scott. New and improved. You know what? I love Phyllis. You know what else? I think she is gorgeous. I think she is incredibly, incredibly attractive person. [to Phyllis] C'mere, c'mere, c'mon! Come on! Come on.

Phyllis: Michael! Come on!

Michael: Oooh!

Phyllis: You don't have to worry. I'm not going to...

Michael: I'm not worried.

Phyllis: ...report you to HR.

Michael: You know what? The only thing I'm worried about... is getting a boner. Good work today, everybody.
Michael: Times have changed a little. And even though we're still a family here at Dunder-Mifflin, families grow. And at some point, the daddy can't take a bath with the kids anymore. I am Upper Management. And it would be inappropriate for me to take a bath with Pam. As much as I might want to.
Pam: He said what?
Michael: I'm an early bird, and I'm a night owl. So I'm wise, and I have worms. Oh, breakfast.
Ryan: I got your sausage, egg and cheese biscuit.

Michael: Yummy, yummy. Thank you, Ryan.

Ryan: What was the thing, ah, you needed me to come in early for?

Michael: Um. The sausage, egg, and cheese biscuit. But thank you. And why don't you take a couple hours. The office is yours. "Home Alone," "Risky Business." Take your pants off, run around. Whatever you gotta do.

Ryan: I'm just going to take a nap in my car until work starts.

Michael: Ok. [Removes biscuit, leaving only sausage, egg and cheese.] Healthier. Gotta watch those carbs.
Michael: Today, I, Michael Scott, am becoming a homeowner. Investing in real estate.
Dwight: Diversifying. Smart.

Michael: Yes it is. Yes it is. It is very important to own property. Back in olden days, they would not even let you vote unless you owned property and they'd throw you in the stocks and humiliate you.

Dwight: And it worked. They should bring the stocks back. People'd obey the law, there'd be less troublemakers.

Michael: Maybe.
Jim: [looks bored. Taps finger on desk. Head falls to desk]
Pam: [laughs]
Pam: Every so often, Jim dies of boredom. I think today it was the expense reports that did him in. And our deal is that, it's up to me to revive him.
Pam: You see Dwight's coffee mug?
Jim: Mm-hmm.

Pam: Sometimes when he's not here, I try to throw stuff in it.

Jim: No way. Let's do this [crumples post it and throws into mug. Misses.] Oh.

Pam: Here.

Jim: Wind.

Pam: Try paperclips. Oh wait. This message. For Dwight.

Jim: Perfect. [misses]

Pam: Oh.

Jim: Oh.
Dwight: You should go.
Michael: Yes. Yes. Final walkthrough.

Dwight: Uh huh.

Michael: Sign the papers at the condo.

Dwight: You have your lawyer there?

Michael: Uh, I don't need one.

Dwight: Can I be your representative?

Michael: I don't need a representative.

Dwight: I think I should be there.

Michael: No, No.

Dwight: I'm good. I can make sure things are up to code.

Michael: No. Dwight. I'm fine.

Dwight: Please, I'm always the guy you rely on at work.

Michael: Well, this isn't about work. This is closing on a condo, it's completely personal.

Dwight: So you're taking a personal day?

Michael: Except that, this is about my living arrangement, and as boss, I need to have a living arrangement in order to do work.

Dwight: Please, I'll make you proud.

Michael: Ok. Fine. Yes, you can come.

Dwight: Yes! As your representative?

Michael: As my associate.

Dwight: Same thing.

Michael: No it is not.
Dwight: I have been Michael's #2 guy for about 5 years. And we make a great team. We're like one of those classic famous teams. He's like Mozart, and I'm like Mozart's friend. No. I'm like Butch Cassidy, and Michael is like Mozart. You try and hurt Mozart; you're going to get a b*llet in your head courtesy of Butch Cassidy.
Michael: Oh, most honorable Pamera. Not offensive, because that's the way they talk in movies.
Pam: You headed out?

Michael: We are. Dwight and I are going to the big thing. So why don't you have everybody work on their expense reports and I'd like them in by the end of the day.

Pam: Ok.

Michael: Very good.

Pam: Have a great time.

Michael: We will. Um, did you do the thing I asked you to do about the magazines?

Pam: Yeah, I changed them to your new address.

Michael: Good. The Small Business Man?

Pam: Yup.

Michael: Maxim? American Way? Cracked?

Pam: Yes, I changed your Cracked magazine subscription.

Michael: How about, uh, Fine Arts? Aficionado Monthly?

Pam: [shakes head]

Michael: NO, well can you get on that, because I don't just read Cracked. Thank you.

Pam: Yeah.

Michael: Ok. See you soon.
Dwight: What kind of shocks you got on this baby?
Michael: I don't know, regular. Normal ones. Nothing fancy. Not my style. What are you doing?

Dwight: [tries to open sun roof] I want to put the top down.

Michael: What? No, Dwight. It's fifty degrees outside. Don't... please...

Dwight: But then no one can see us.

Michael: I... Just... Would you put it up? [roof opens] Ok. Fine. Just leave it down. Whiner.

Dwight: Check it out. [points at sunglasses] Terminator.

Michael: I do not understand what you spend your money on.
Kevin: [paper football lands on desk] Ooh.
Jim: Hey, Oscar, on these new expense reports, do we really have to go back to last quarter?

Oscar: Yeah. It's a terrible system, I know.

Jim: [points at paper on desk] What does 2005 season mean?

Oscar: Eh.

Jim: Wait a minute, what is this?

Oscar: It's a scoreboard.

Jim: What?

Oscar: Kevin and I play this paper football game when Michael's out.

Jim: Really?

Oscar: Yeah.

Kevin: Or when we're bored.

Jim: Oh my God! Wait, this goes back two years.

Kevin: We're bored a lot.
Jim: [flicks football onto Kevin's desk] OH!
Kevin: Oh!

Oscar: Sweet!

Jim: Yes! So close. I really love the paper triangle flicking and hitting things game. Yeah.

Kevin: We call it Hate Ball.

Jim: Why?

Kevin: Because of how much Angela hates it.

Jim: Hey, do you guys have any other games?

Kevin: Sometimes we play "Who can put the most M&M's in their mouth?"

Angela: You play that.

Oscar: You should ask Toby to teach you Dunderball.
Michael: Home, sweet home.
Dwight: Which one's yours?

Michael: Right there. My sanctuary. My party pad. Someday I can just see my grandkids learning how to walk out here. Hang a swing from this tree. Push them back... wait... [turns around] no, it's this one, right here. Home, sweet home.
Jim: [bounces ball off wall with Toby] So that's what this sound is all day.
Carol: Michael, this is Bill. He's the head of the condo association.
Michael: Oh, how are you? Nice to meet you, Bill. Bill. Mr. Bill. OHHH NOOO. MR. BILL. OHHH! SNL? When they pull him apart? He'd always get rolled over by something.

Bill: Nice to meet you.

Michael: Nice to meet you too.

Dwight: This is smaller than your old place.

Michael: Yeah, small. I'm buying it and I'm not renting it. So, it's still an upgrade. He doesn't know anything about property ownership. Kind of an idiot. Um.
Dwight: Actually, I do own property. My grandfather left me a 60 acre working beet farm. I run it with my cousin, Mose. We sell beets to local stores and restaurants. It's a nice little farm. Sometimes teenagers use it for sex.
Carol: Are we ready to sign some papers?
Dwight: Actually, no. We have a couple of questions, about the neighborhood.

Bill: It's very safe. It's very clean. Also, it's very accepting of all lifestyles.

Carol: It's a very gay-friendly neighborhood.

Michael: Oh. Good. That's good. It's good to be accommodating of that.

Dwight: Let's go check out the master bedroom.
Jim: Stanley. Just played Dunder Ball with Toby. What about you, you got any games?
Stanley: Yeah, I got a game. It's called "work hard so my kids can go to college."

Jim: Fair enough.
Michael: This, my friends, is the master bedroom. Check out the cathedral ceilings. Those are like seventeen feet high. We have cable readiness. [points at wall] Right there. I am going to totally pimp this place out. I am going to put a surround sound system. I am going to put a plasma screen right against this wall.
Dwight: Oh. Terrible idea.

Michael: I'm putting my bed right over here.

Dwight: No, no, no, no, no. This is a shared wall. Neighbor throws his wife into the wall, plasma screen hits the floor. Totally smashed.

Michael: Well, then I will get a warrantee.

Dwight: Warrantees don't cover it, plus they're a rip-off.

Michael: Well then I won't get a warrantee.

Dwight: Shh Shh.

Michael: So that's the problem, is solved. What?

Dwight: Listen. [puts ear to wall] Can you hear that? Oh man. These babies are thin.
Jim: [sings Olympic theme song] This scented candle ...andle ...andle. Which I found in the men's bathroom ...room ...room. Represents the eternal burning of competition. Or something.
Kevin: It smells like cookies.

Jim: Yes it does. Yes it does my friend. Ok, we will be competing for gold, silver and bronze yogurt lids.

Pam: Now the bronze are really blue, and they're also the back side of the gold, so no flipping. K? Honor system.
Angela: I do play games. I sing and I dangle things in front of my cats. I play lots of games. Just not at work.
Jim: Let the games begin. [sings Olympic theme]
Carol: And then, I just need you to sign here at this arrow.
Dwight: What kind of mortgage did you get?

Michael: Uh... Ten year.

Carol: Well, ten over thirty, so thirty year total.

Michael: What? Wha? You said ten.

Carol: Ten year fixed, over thirty. Thirty year total.

Dwight: Ho, thirty years.

Michael: Ok, ok, ok.

Dwight: Wow, you'll be paying this off in your mid-seventies.

Michael: Alright.

Dwight: Forget about retiring when you're 65. Hey, I've got an idea. You know that extra bedroom? If the whole girlfriend thing never happens, that's where the nurse can live.

Michael: Ok. Alright. Oh boy.

Dwight: Well, this is it.

Carol: Whenever you're ready.

Michael: Um. Oh. [moves stove burner] Oh, ok. Is that suppose to come off?

Carol: Actually yes.

Dwight: Hey, look! Cool. Carpenter ants.

Michael: Um. I'm going to take a little breather for a second. Excuse me.

Dwight: We'll be here waiting for you.

Michael: Oh, man.
Dwight: A thirty year mortgage at Michael's age essentially means that he's buying a coffin. If I were buying my coffin, I would get one with thicker walls so you couldn't hear the other dead people.
Carol: Whenever you're ready, Michael.
Michael: Uh. [breathes deeply, head at knees]
Jim: You have what is the national sport of Icelandic paper companies. And I'm blanking on the name, can you help me out Pam?
Pam: Jim, they refer to it as Flonkerton.

Jim: Hum.

Pam: In English, box of paper snowshoe racing.

Jim: Fair enough, but I like Flonkerton.
Pam: The thing about Jim, is when he's excited about something, like the Office Olympics, he gets really into it and he does a really great job. But the problem with Jim is that he works here, so that hardly ever happens.
Jim: So, who will be challenging Kevin in Flonkerton? Anyone?
Phyllis: I'll do it.

Jim: Yes! Phyllis! [claps] Phyllis, just put your foot right through here [lifts strap on box of paper]. Right through the flonk.
Michael: The ceilings are lower than they were last week. That, I don't... I don't...
Carol: What?

Michael: ...know if you showed me this same unit or not.

Carol: Michael, this is the unit you saw and...

Michael: Where are all the hot people? I was told that there would be all these attractive singles.

Carol: Who told you that?

Michael: As far as I can tell, I'm the best-looking person here.
Michael: There's a basic principle in real estate, that you should never be the best-looking person in the development. It's just sorta common sense, because if you are, then you've no place to go but down.
Carol: Is this a financial thing? If it's a financial thing, what some people do is they rent out the third bedroom.
Michael: No, no, no.

Carol: That's some extra income for you.

Michael: I am not going to rent the third bedroom. I want a price reduction or I am a-walkin.

Carol: You will lose $7,000 if you walk away right now.

Michael: Ehhhh....
Michael: I made the right decision. I'm glad I signed. I'm a homeowner. Right? Good to be a homeowner. Diversifying. This is good. This is fun. We're having fun.
Dwight: Totally having fun. Can you imagine those poor saps stuck at the office today? [laughs]
Jim: Here we go. Here we go.
Pam: Go! Go! Go!

Oscar: Pair of shoes!

Jim: Dig deep, dig deep! OHHHhhh! It's Phyllis!

Pam: It's Phyllis!

Jim: Phyllis by a nose. Gold medal in Flernenton.

Pam: Flonkerton.

Jim: Thank you, delegate from Iceland.

Meredith: Wow!
Kevin: [empties bowl of M&M's into his mouth]
Jim: Wow! Ok. No one else should even try! Gold medals! Give him medals. Wow.
Michael: There's something else Dwight wanted to talk to you about. I have a surprise for you, for helping me out today.
Dwight: You didn't have to...

Michael: No, no. I insist. I insist. Because you've really done some great work. Great work. And that is why, I am going to let you move into my third bedroom and pay me rent.
Michael: Why did I do it? Because I believe in rewarding people for their efforts. Ah. I rewarded Dwight with the room, and he is rewarding me back, ah, with $500 plus utilities.
Dwight: I don't even know what to say.
Michael: I'm thinking, lock into a four year commitment, we'll go month to month after that. Or, until I start dating, have a girlfriend, then you're, you know, you're gone.

Dwight: Question. Where can I put my terrarium?

Michael: What the hell is a terrarium?

Dwight: It's a fish t*nk for snakes and lizards.

Michael: Oh, so an aquarium. Ah, that will not come into this place.

Dwight: Question. My grandparents left me a large number of armoires.
Pam: Are you sure you don't want to play?
Angela: I'm sure.

Pam: Come on Angela, don't you have a game?

Angela: I have one, yes.

Pam: Well, let's play, what is it?

Angela: I call it Pam Pong. I count how many times Jim gets up from his desk and goes to reception to talk to you.

Pam: We're friends.

Angela: Apparently.

Jim: Very nicely done. I think that's H-O-R for Stanley, and H-O for Phyllis.

Phyllis: Are you calling me a ho?

Jim: Oh my god. Phyllis, coming alive. I like it.
Dwight: Question. What about carpooling, who pays for the gas?
Michael: We take separate cars.

Dwight: Question. Can sometimes I drive your car and you drive mine?

Michael: Why would we do that?

Dwight: Just for fun?

Michael: No.

Dwight: Question. Who is the primary on the fire insurance?

Michael: EHHHHNT. Game over. Offer revoked. Dwight. I'm sorry, but you reach out and you try to be a nice guy, and help out a friend, and this is what happens. This is what I get. Oh god. I'm ... Ok.
Dwight: Thank god. It was nice of him to offer, but I live in a nine bedroom farm house. I have my own crossbow range. It's a perfect situation for me. Although two bathrooms would have been nice, we just have the one. And it's under the porch.
Oscar: Ah...
Everyone: OH!

Jim: Who had someone from Vance Refrigeration?

Ryan: I did.

Jim: Ryan Howard. Ryan! [claps] Gold medal.
Pam: I made something for our closing ceremonies.
Jim: What? [looks in box] Oh my god. Where did you have time to make that?

Pam: a*t*matic voicemail.

Jim: Alright Pam, alright [gives her hi-five]. Nice work!

Pam: [sees Angela making check mark on tally sheet]

Stanley: A little bit more and I would have had it.
Dwight: You know you can always refinance your mortgage. We had a 15 year on our beet farm. We paid it off early.
Michael: Yeah, well, you know what? Nobody cares about your stupid beet farm. Beets are the worst.

Dwight: People love beets.

Michael: Nobody likes beets.

Dwight: Everybody loves beets.

Michael: Nobody likes beets, Dwight. Why don't you grow something that everybody does like? You should grow candy. I'd love a piece of candy right now. Not a beet.

Dwight: Let's get this roof going.

Michael: Stop it! [smacks Dwight's arm]

Dwight: Ow.
Jim: Final lap. Final lap. Time to b*at is 1:15.
Stanley: Oscar!

Crowd: Go! Go! Go!

Jim: Time to b*at is one minute, 15 seconds. Here they come. [Michael and Dwight enter] Guys?

Dwight: What is going on?

Jim: Nothing. Guys? Timer's still going? Er?

Dwight: That's my stopwatch.
Jim: [hands expense report to Oscar] Here you go. All done.
Oscar: Great.
Jim: Yeah, I filled out the expense reports. That took about five minutes and then I closed two sales at lunch time. So, about as productive as any other day. If not more so.
Ryan: I figured I could throw it away now, or I could keep it for a couple of months and then throw it away. I mean, it was really nice of Pam to make them, but what am I going to do with a gold medal made of paper clips and an old yogurt lid?
Jim: Hey.
Pam: I have 59 voicemails.

Jim: Mmm. Hey, can you ignore those and do something for me instead?

Pam: Sure.

Jim: Okay, today. 5 o'clock. Closing Ceremonies.

Pam: Really?

Jim: Notify the athletes.

Pam: Cool.
Jim: Michael.
Michael: Yeah. Jim. Slim Jim. What's going... What's going on?

Jim: Nothing. I just wanted to congratulate you on your condo.

Michael: Oh. Thanks. Thanks. It's very cool. It's a three bedroom, gay-friendly.

Jim: Nice.

Michael: You know.

Jim: Hey, would you mind coming out here for a second? I just have something for you.

Michael: Really?
Michael: What's this?
Jim: These are the Closing Ceremonies. Step up. You're on the top one. [Michael stands on podium] Congratulations to Michael, because he closed on his condo. So, gold medal. [everyone claps]

Michael: I don't really know what to say. Um, I'm not one for making speeches, but ah, my heart is very full at this moment.

Jim: And for Dwight Schrute, the silver medal.

Michael: Get up here, Dwight.

Dwight: Silver medal.

Michael: Yep, not as good as gold. [national anthem plays] Why are you playing the national anthem?

Jim: Um... 'Cause your condo's in America.

Michael: Oh. [doves move across cord] What the hell is that?

Jim: Those are the doves.
Pam: Dunder Mifflin, this is Pam. Sure, can I ask who's calling? Just a second.
Jim: Jim Halpert. What? How did you get this number? Stalker.
Pam: Katy and Jim met in the office. And now I guess they're like going out, or dating, or something. And, uh... I don't know! You know? They're just... She calls him, and they... You know, I'm sorry. I feel like I'm talking really loud. Am I talking really loud?
Jim: So we're still on for lunch? You're meeting me here? Okay. Great. Bye.
Pam: [to Jim] Hey! You can just give her your extension.

Jim: Okay.
Michael: Howard, slash Ryan, Ryan Howard is sitting in my office. And he has been a temp here for a couple of months and he's kind of gotten the lay of the land a little bit. Had a few laughs along the way. And now he wants to know what I think.
Ryan: The temp agency wants to know what you think.

Michael: Shall we? Let us proceed. First up, proficiency in necessary skills. Aaaaeeexcellent! [laughs]
Dwight: Michael's in there right now evaluating the temp. He hasn't evaluated me in years.
Michael: Five years from now, what do you want to do? Where do you want to be?
Ryan: Ah, well, I'm interested in business.

Michael: Oh! Good. Ambitious. Excellent. Want to be a manager?

Ryan: Uh, no, actually, uh, what I want is to own my own company.

Michael: That is ridiculous.
Michael: Ryan's about to attend the Michael Scott School of Business. I'm like Mr. Miyagi and Yoda rolled into one.
Michael: [speaks in a Yoda voice] Much advice you seek. [regular voice] Do you know who that is?
Ryan: Fozzie bear?

Michael: Mmm... No. That was Yoda.
Michael: There are ten rules of business that you need to learn. Number one[/b]: You need to play to win. But... you also have to win to play.
Ryan: Got it.

Michael: And I will give you the rest of the ten at lunch.
Michael: [to Ryan] [makes clicking noises like sh**ting a g*n] Hey!
Dwight: Michael and I have a very special connection. He's like Batman, I'm like Robin. He's like the Lone Ranger, and I'm like Tonto. And it's not like there was the Lone Ranger, and Tonto, and Bonto.
Oscar: [in background, on phone] But it says no late fee... .
Dwight: [alarm sounds] People!

Angela: Okay! Everybody!

Dwight: This is not a test! Move to the exits!

Angela: Do not panic!

Dwight: Head towards the exits.

Angela: Safety partners.

Dwight: Get up off your desks!

Angela: Do not panic.

Oscar: [in phone] No, I don't hear it? Alright.

Dwight: No, panic is warranted!

Angela: Go in single file lines.

Oscar: [in phone] No, no. Finish the...

Dwight: This is not at drill!

Angela: Arms at your sides! Arms at your sides!

Dwight: Please, move quickly! This is a paper company, people! Step lively!

Angela: Go, let's go.

Dwight: This whole place is a tinder box, it is ready to blow!
Dwight: This is not a test! Can you leave?!
Phyllis: Oh, you say that every time.

Dwight: DO YOU WANT TO DIE?

Phyllis: Oh, boy...

Dwight: Do you want to die? OUT!!

Angela: Alright, let's go, let's go.

Dwight: STANLEY! Have you ever seen a burn victim?

Angela: Come on, you're safety partners!

Dwight: Move to the exits!

Angela: You're safety partners!

Dwight: We've got smoke! We've got smoke! Smoke! Gah! [Spots Kelly] Oh, Kelly! You're okay! I've got you!

Kelly: I'm okay!

Dwight: Cover your nose and mouth. Breathe through your nose.

Kelly: Let go of me!

Dwight: Breathe through your nose. Remove your stockings. Okay? They'll melt right into your flesh! Stay below the smoke line. Let's go! Clear out, stat! STAT MEANS NOW!
Michael: Yes, I was the first one out. And, yes, I've heard women and children first. But, we do not employ children. We are not a sweat shop. Thankfully. And, uh, women are equal in the workplace by law. So, I let them out first, I have a lawsuit on my hands.
Michael: Another rule of business is being able to adapt to different situations.
Ryan: Yeah.

Michael: Adapt. React. Re-adapt. Act. All right? That's rule number two.
Dwight: Okay, guys, listen up, we need a head count. We need to count off. Michael's number one. Where is he? Where is he?
Michael: So what was rule two?
Ryan: Ah... adapt, react, re-adapt, act.

Michael: Okay, well, let's... . let's kind of take it a little slower.

Dwight: Hey, Michael. Um... Ryan needs his number for the count off.

Michael: Okay, uh, well, one is taken.

Ryan: Uh, okay, two?

Dwight: NO!

Ryan: Okay... uh, sorry?
Dwight: Okay, he can have 14. Marjory's not here today.
Michael: Well, he needs a permanent number, right?

Ryan: No.
Ryan: ...I don't.
Dwight: Oh, you know what else? I thought of a nickname for the three of us. Three Musketeers.

Michael: Um, yeah. Okay. That... No, no, no. I got one. I got one. The Three Stooges.

Dwight: That's funny, too. But if we're the Three Musketeers...
Ryan: I don't want to be like "a guy" here. You know? Like, Stanley is the "crossword puzzle guy". And Angela has cats. I don't want to have a thing... here. You know, I don't want to be the "something guy".
Jim: Okay, you know what? I am going to be, uh, setting the agenda here. Okay? Can everybody gather up, please? Important announcement. Very important announcement. I think this is a perfect opportunity for all of us to participate in some really intense, psychologically revealing conversations. So we're going to be playing Desert Island, umm, Who Would You Do?
Stanley: Ooh.

Jim: And, um...

Pam: ...Would You Rather?

Jim: Would You Rather. Would You Rather is our third game.
Dwight: [to firemen] Hey guys, great response time. Listen up, I got some theories. Okay, there's a...
Jim: Okay, so... three books on a desert island? Angela.
Angela: The Bible.

Stanley: That's one book. You've got two others.

Angela: A Purpose Driven Life.

Jim: Nice. Third book?

Angela: No.

Jim: Okay. Phyllis.

Phyllis: Um, The DaVinci Code.

Angela: The DaVinci Code!

Jim: Nice.

Angela: I would take The DaVinci Code... so I could burn The DaVinci Code.

Dwight: Okay. Great, that's going to keep you warm for like 7 seconds. Question[/b]: is there fire wood on the island?

Jim: I guess.

Dwight: Then I would bring an axe, no books.

Jim: Uh, it has to be a book, Dwight.

Dwight: Fine. Physician's Desk Reference.

Jim: Nice. Smart.

Dwight: ...hollowed out, inside[/b]: waterproof matches, iodine tablets, beet seeds, protein bars, NASA blanket, and, in case I get bored, Harry Potter and Sorcerer's Stone. No, Harry Potter and the Prisoner of Azkaban. Question[/b]: did my shoes come off in the plane crash?
Michael: Rule number four. In business, image is everything - Andre Agassi. This car is an investment. Right? If I have to take out a client or I'm seen around Scranton in it. I love it. I love this car. Do you like it?
Ryan: Yeah.
Jim: Okay. Thought people read more books.
Jim: DVDs! Five movies. What would you bring to the island? Yes! Meredith?
Meredith: Legends of the Fall, My Big Fat Greek Wedding, Legally Blond, Bridges of Madison County...
Jim: Wow.
Pam: Legends of the Fall?

Jim: Wow. Bridges of Madison County, Legally Blond, these movies are just... .

Pam: Well, I kind of liked Legal...

Jim: Wait, wait, wait. Pam. No. Do you understand? The... the game is Desert Island Movies, not guilty pleasure movies. Desert Island Movies are the movies you're going to watch for the rest of your life! Forever! Unforgivable.

Pam: I take it back.

Jim: Unforgivable.

Pam: I take it back!

Jim: Good.
Meredith: ...and Ghost. But, ah, just that one scene...
Dwight: Is this your car, Ryan?
Michael: Wow, some pretty big books back there, huh?

Ryan: [to Dwight] Don't...

Dwight: Good shocks.

Michael: Hello, Mr. Egghead! Woop! So... oh, Stanley Kaplan! I know him. 'M' is for m*rder, 'P' is for...

Ryan: That's actually a test prep book.

Michael: ...for Phone. What?

Ryan: That's a test prep for business school.

Michael: Um, oh, thinking about business school?

Ryan: I just got in. I applied, I go at night.

Michael: Really?

Ryan: Yeah.

Michael: So you think you know a lot about business?

Ryan: No, not yet.

Michael: Uh huh.

Ryan: Just started.

Michael: Yeah. Quiz me.

Ryan: I... wouldn't even know where to start.

Michael: Come on, egghead. Let's do it.

Dwight: Do it.

Michael: Quiz me up.

Ryan: All right, um... Why have people been rethinking the Microsoft model in the past few years?

Michael: Uh...
Michael: When I was Ryan's age, I worked in a fast food restaurant, to save up money for school. And then I spe... lost it in a pyramid scheme. But I learned more about business, right then and there, than business school would ever teach me, or Ryan would ever teach me.
Ryan: Is it cheaper to sign a new customer? Or to keep an existing customer?
Dwight: Keep an existing...

Michael: [to Dwight] Shut, it. Can I... can I just do it please? [to Ryan] Uh, it's equal.

Ryan: It is ten times more expensive to sign a new customer.

Michael: Okay. Yes! It was a trick question.

Dwight: Yeah, but look, I mean, he didn't need business school. Okay, Michael comes from the school of hard knocks.

Michael: Okay, Dwight.

Dwight: Self taught. You didn't even go to college.

Michael: You know what, Dwight? You don't need to help me here. Okay? Well, you know... Maybe you should go to business school like Ryan, then... then you'd know what you're talking about.

Dwight: [scoffs] Come on. I'm studying with the master, huh?

Michael: For instance, why don't you go to business...

Dwight: [to Ryan] You should learn from him, right?

Ryan: I am.

Dwight: Right?

Ryan: I am.

Michael: Stop. Dwight. You know what? You're acting like a dork. Would you cool it? Please. Okay. Hey! He's not your five year old brother, Dwight. He's a valued member of this company... and you know what? He knows more about business than you ever will.

Dwight: Stupid!
Michael: I did not go to business school. You know who else didn't go to business school? LeBron James, Tracy McGrady, Kobe Bryant. They went right from high school to the NBA. So... so it's not the same thing. At all.
Michael: Look at this stuff. Market fragments. What is that supposed to be?
Ryan: It's a way of looking at consumers as subsets of a larger client base.

Michael: You are so smart. You are so eff-in' smart. You should be teaching me.
Jim: Pam? Get us back into it.
Pam: Okay.

Jim: Five movies. Go ahead.

Pam: Um, Fargo, um, Edward Scissorhands, Dazed and Confused...

Jim: Ooh, definitely in my top five.

Pam: Yes. In my top three, so suck it.

Jim: What?

Pam: Breakfast Club. Um... The Princess Bride and...

Jim: Okay that's five.

Pam: No, my all time favorite!

Jim: Pam, play by the rules.

Pam: All time favorite.

Jim: Play by the rules. Dwight. All time favorite movie.

Dwight: The Crow.
Michael: I became a salesman... because of people, I love making friends. But then I was promoted to manager, at a very young age. I still try to be a friend first, but... You know? I'm very successful... your coworkers look at you differently. Huu, what do you think?
Ryan: Maybe we should get some air.

Michael: Nah, I'm okay.

Ryan: I'm really uncomfortable.
Jim: All right. Let's move on. Let's move on to the main event. Who Would You Do?
Kevin: Present company excluded?

Jim: Um, not neccessari...

Kevin: Pam.

Oscar: Pam.

Jim: Um... okay. Ah, you know what? Maybe I'll... I'll finish explaining the rules. Let's... let me explain it first, and then...

song: ["Everybody Hurts" by R.E.M.] Think you've had too much / in this life.

Jim: Yeah, so we'll get right... You know what? I'll be right back. Stanley, you're taking over for me, buddy. I'll be right back.

Stanley: Okay, um...
Jim: Dwight. Dwight.
Song: Everybody hurts,

Jim: Come on Dwight! Use words.

Song: Sometim... .

Dwight: Why didn't I go to business school?

Jim: Who goes to business school?

Dwight: The temp.

Jim: He does?

Dwight: Yeah, it's all him and Michael talk about anymore.

Pam: You know, I bet Ryan thinks to himself 'I wish I were a volunteer sheriff on the weekends'.

Dwight: He doesn't even know that I do that.

Pam: You should tell him.

Dwight: Oh yeah, Pam. Right. That's going to help things, just talk it out. I hope the w*r goes on forever and Ryan gets drafted.

Pam: Dwight.

Jim: What?

Dwight: I'm sorry I said that, I didn't... just part of me meant it. Besides, he'd end up being a hero anyway.

Jim: You know what you should do? You should quit. And then, that would stick it to both of them.

Dwight: Oh Jim, I'm not going to quit. Then Ryan wins.

Jim: Yeah. You're right.

Dwight: Thanks you guys. I just need some alone time.

Pam: Kay.

Song: Everybody hurts

Jim: Alright buddy.

Song: Everybody cries

Roy: Hey! Guys, what's going on?

Jim: Nothing.

Pam: Hey!

Song: Everybody hurts

Roy: What's up? Can I hang out with you guys for a bit?

Song: Sometimes

Roy: The warehouse guys are a bunch of jackasses sometimes.
Stanley: Come on people, you know the rules of the game now.
Michael: Oh, hey. Game, what game are we playing here?

Stanley: Okay. It's called Who Would You Do?

Michael: Oh, I play this at home all the time while I'm falling asleep. What, uh... . Where are we? Where are we here? Mmm.. Roy? Roy? Who would you do, Roy?

Roy: Uh... Oh, I got it! Uh, what's the name of that, uh, tight ass, uh, Christian, uh, chick. The, uh, the blond?

Angela: My name is Angela.

Roy: Hey, Angela! Roy. Nice to meet you.

Michael: Aaaall right. Who's next, who's next, who's next, who's? Jim? You're next. Who would you do?

Jim: Um... Kevin, hands down. Yeah. He's really got that teddy bear thing going on, and afterwards, we could just watch bowling.

Michael: Well, I would definitely have sex with Ryan. 'Cause he is going to own his own business.

Roy: You're all gay.

Michael: Who's, uh... Who's next? Who we got? Whooo...

Ryan: [answers phone] Hey, no, I can talk, I can talk, I can talk... this is great timing.

Michael: Wish I had my cell phone, but I left it inside. So...

Dwight: Would that make you happy?

Michael: What's that?

Dwight: If you had your cell phone, it would make you happy?

Michael: Yeah.

Dwight: I'm on it.

Michael: Dwight. Hey!

Angela: You can't go in yet!

Michael: Dwight, don't! He is an idiot. The man is an idiot, ladies and gentlemen.

Kevin: What if he dies in the fire? And that's the last thing you ever said to him.

Michael: I didn't say it to him. I said it about him.
Meredith: ...Jim.
Phyllis: Definitely Jim.

Kelly: Definitely, definitely, Jim.

Phyllis: Come on, Pam.

Kelly: How about you Pam?

Pam: Um... Oscar's kind of cute.

Phyllis: Yeah, I like Oscar.

Pam: Ooh, Toby!

Michael: [in the background] How long does it take to find a cell phone? I don't know either.

Meredith: Is there anybody else.

Kevin: [clears his throat]
Jim: [on the phone] Hey, where are you? Oh good. Yeah. We're just here, we're playing Desert Island. It's when you pick your five favorite DVDs...
Michael: Seriously, where the hell is Dwight? Hey, call my cell phone. It'll make it easier for him to find.

Ryan: What's your number?

Michael: I gave it to you in the car.

Ryan: Um...

Michael: I saw you program it in.

Ryan: You got to... you got to give it to me again.

Michael: Okay. Alright.

Ryan: Now I have it.

Michael: Uh, I better tell somebody. [to fireman] Excuse me, sir...

Dwight: [coughing]

Michael: Dwight!? Great goin'. God, Man! Why did you go in there? What... Everybody was scared out of their wits, man? Oooh.

Dwight: [coughing] Everyone, okay? Uh, I have an announcement. Apparently, in business school, they don't teach you how to operate a toaster oven. Because some smart, sexy temp left his cheese pita on oven instead of timing it for the toaster thing.

Michael: Wow. Okay. Well, I guess they don't teach how to operate a toaster oven in business school.

Dwight: That's exactly what I said.

Michael: Hey, did you miss that day there, Ryan?

Dwight: Were you absent?

Michael: Toaster Oven 101?

Dwight: You failed?

Ryan: I am so sorry.
Michael: Hey! I know what'll impress everybody, I'll start a fire. Oh, man. Bad idea. Bad idea jeans.
Dwight: I have a song. Attention, everyone! That I want to sing. That I wrote especially for this occasion when I was up there among the flames. Ready? [sings to Billy Joel's "We Didn't Start the Fire"] Ryan started the fire! It was always burning since the world's been turning!
Dwight and Michael: [singing] Ryan started the fire! It was always burning---

Dwight: Everybody!

Michael: [singing] ...since the world was turning.
Ryan: I can't believe I started the fire.
Dwight and Michael: [sing gibberish to "We Didn't Start the Fire"]
Dwight: [singing] ... Marilyn Monroe!

Dwight and Michael: [singing] Ryan started the fire! It was always burning...

Dwight: Eat it! You gotta eat it. You have to eat it!
Katy: Hi!
Jim: Hey.

Katy: How are you?

Jim: Good, how are you?

Katy: I'm good. It's good to see you.

Jim: Good to see you, too.

Katy: I'm hungry.

Jim: Yeah, I am too.

Katy: Oh, I have been thinking the whole way over and I have my answers.

Jim: What answers?

Katy: Um, for the... the desert island.

Jim: Oh! Right! Right, right, right, come-ah on, on, on. [to everyone] Ladies and gentlemen! Gather around! We have one more participant. Come on, be polite. Be polite. [to Katy] Desert Island. Five movies. Go.

Katy: Okay, um, first, Legally Blond.
Pam: I forgot what a super, nice girl Katy is. And just... good for Jim! They are so cute together. And, um, what an adorable car.
Jim: Okay, I think the game's over... People are like leaving. There was a bigger crowd last time. Do you just want to go to lunch?
Katy: Okay.

Jim: Yeah?

Katy: Alright! You want to drive?

Jim: Sure.

Katy: Alright.
Katy: [looking at Roy and Pam] They are soo cute.
Ryan: I'm really sorry, Dwight.
Dwight: Answer me this, though.

Ryan: What?

Dwight: Was it worth it? Was it worth it temp?

Ryan: No.

Kevin: Was it worth it?

Dwight: Really?

Ryan: I'm really sorry, Dwight.

Dwight: The fire guy! The fire guy!
Dwight: [sings] Joe McCarthy, Richard Nixon, Studebaker, television, North Korea, South Korea, Marilyn Monroe, Ryan started the fire!
Michael: Okay. Rule five - safety first, i.e. don't burn the building down. Okay? That should be a no brainer.
Michael: Oh... look! Ryan is book smart. And I am street smart. And book smart.
Michael: I'll give you the rest of the ten tomorrow.
Michael: Happy Halloween, everyone! [notices Pam, in her cat costume] Oh... that's great!
Pam: Hey... Happy Halloween. Jan called.

Michael: Ohh... OK.
Michael: I know why she's calling. It's the end of the month, and I was supposed to let somebody go by the end of the month. And somehow I'm supposed to put on a costume and smile. [dials a number on his speaker phone] Okay.
Sherri: [on phone] Jan Levinson's office.

Michael: Hey, Sherri. Michael Scott returning.

Sherri: Oh, she's in a meeting. Uh, she just wanted the name of the employee you let go.

Michael: Well, I'm gonna wait till the end of the day. Because the book said it's best to wait till the end of the day.

Sherri: I just need the name of who you're planning to let go.

Michael: I don't know... yet. I will have to call her back.

Sherri: I know she wanted the name.

Michael: Okay... Sherri?

Sherri: Yeah?

Michael: If you were getting fired, how would you wanna be told so you could still be friends with the person firing you?

Sherri: Jan wants the name as soon as possible, Michael.

Michael: Thanks.

Sherri: Mm-Hmm.

Michael: I'll call her back. [talks softy, to himself] Wish I could fire Sherri.

Sherri: Hey, I'm still here.

Michael: Okay! I'm sorry.

Sherri: Yeah.

Michael: No?

Sherri: OK.

Michael: Bye.

Sherri: Hanging up now.
Michael: I mean you hear about layoffs in the news, but when you actually have to do it yourself, it is heavy stuff. It's... these are people's lives you're talking about.
Pam: [entering] You wanted me?

Michael: Yes.

Pam: [notices Michael's costume] Papier-mache?

Michael: Yes.

Pam: Hmm.

Michael: Yeeesh.

Pam: Mm-hmm.

Michael: Um, Pam, I have to let somebody go today. This is, uh, the hardest thing I've ever had to do.

Pam: Why did you put it off until Halloween?

Michael: Because it's very scary stuff.

Pam: I think it's gonna put a damper on the party a little.

Michael: You're worried about the party? There's a man's life at stake here.

Pam: So it's a man?

Michael: No. Or a woman. A human life. If you had to guess, who it would be based on their job performance... and who you think deserved to be fired - who would that be?

Pam: I just answer the phone.

Michael: And... sometimes you just let it go to voicemail.

Pam: You're costume is fantastic! [laughs]

Michael: I know. I sent away for it in July from a catalog. [bobs his head around, causing the costume head to jiggle around]

Pam: Oh no, don't, don't, don't, don't. [Michael laughs] Aah! [laughs, then leaves] Okay...

Michael: Oh, man. Okay, I have to fire somebody.
Dwight: [eyeing Jim's costume] What is that?! What are you supposed to be?
Jim: I'm a three hole punch version of Jim. 'Cause you can have me either way. Plain White Jim, or Three-hole Punch.

Phyllis: That's great!

Jim: Oh, yeah.

Dwight: Yeah, well look... [pulls his hood over his head and pops up his light saber] What about me?

Phyllis: What are you? A monk?

Dwight: I am a Sith Lord. [looks at Jim] Oh big deal. Three round pieces of paper taped to a shirt. This cost me 129 dollars.

Phyllis: Ass.
Michael: Hey.
Oscar: Michael.

Michael: You guys excited about the party?

Angela: Yeah.

Michael: It's gonna be fun.

Kevin: Yeah.

Angela: Yes.

Michael: [looks to Oscar] Oh, boy... look at you! Haha. Showing your colors. Bet you wish you wore a dress every day.

Oscar: What are you implying?

Michael: All good. Happy Halloween. What happened to all those spooky decorations that we had? The cobwebs and such?

Angela: You know, I don't know. We put them all up last night.

Michael: Well, you know what? Go buy some more. I'll approve the overages. Sound good?

Angela: Yeah.

Michael: Good. Oh, yeah, also about budget stuff. Um, I'm going to need you to find, like a, a full employee salary, plus benefits, like fifty grand. I'm going to need you to find 50 grand in the numbers.

Angela: But we don't keep two sets of books.

Michael: Well, that's not what I'm saying. Just, you know, find it. Pretend that your jobs depend on it.
Michael: Mmm-hm-hmm. Interesting take on Dorothy. I love it. Hey, you know what would even be better? Soccer ball and cleats.
Kelly: Why is that?

Michael: "Bend It Like Beckham."

Kelly: Oh, like ... the movie about the Indian girl who plays soccer?

Michael: [laughs] Yeah. That would be perfect.

Kelly: Yeah, I mean, I guess I could do that. I don't really play soccer or anything.

Michael: Well, I don't really have two heads. So...
Dwight: Wait, what are you again? Oh, right... Three-hole PUNCH! [punches Jim in the chest and cracks up laughing]
Pam: Okay, greatest strength.
Jim: Okay, okay...

Pam: A dog-like obedience to authority

Jim: Nice.

Pam: But that doesn't sound good.

Jim: Okay, okay. Um, how 'bout, the ultimate team player? [Pam laughs and types]
Jim: Dwight is... special. But, I don't believe that his talents are being used in this office. So Pam and I have put his resume on Monster.com, Google, Craig's List. We're really interested most in jobs that take Dwight out of state. Um, preferably Alaska... or India.
Pam: He's a g*n nut.
Jim: Um. He sticks to his g*n.
Angela: Well, I looked through all the budgets. And there is one department...
Oscar: Yes?

Angela: ... that has three people...

Oscar: Yeah?

Angela: ... doing the work that could be done by two.

Oscar: This is great. [Angela shakes her head] Oh.

Kevin: Yeah. Oh.
Michael: Who do you think it should be?
Dwight: Jim. Definitely.

Michael: No, Jim brings in money.

Dwight: Phyllis.

Michael: Eh.

Dwight: Stanley. Pam. Oscar. Meredith. Kevin. Angela.
Michael: It's not a popularity contest. Although it does make sense to fire the least popular because it has the least effect on morale.
Dwight: One of the warehouse guys.
Michael: [turns to the fake head, listening] What? There was someone left off that list? Who?

Dwight: Who is he saying?

Michael: You're right, I didn't even think of him.

Dwight: No, Michael.

Michael: Yeah, that's actually a really good idea.

Dwight: No, not me.

Michael: Yeah... I could.

Dwight: Not Dwight.

Michael: I'm not saying that's what he said.

Dwight: I know that's what he said.

Michael: [listening to his head] What?

Dwight: Tell him, not Dwight.

Michael: That is not a very nice thing to say about him.

Dwight: Tell him to stop.

Michael: Are you kidding?

Dwight: Quiet, you.

Michael: I agree. He'd land on his feet.

Dwight: Make him be quiet.
Angela: Those aren't chips and dip.
Pam: No, I made brownies.

Angela: Uh!

Pam: ... What?

Angela: I'm just trying to figure out why you're sabotaging things.

Pam: I made brownies.

Angela: And I made cookies. Same category.
Pam: I'm guessing Angela's the one in the neighborhood who gives the trick-or-treaters some toothbrushes. Pennies. Walnuts.
Pam: [on phone] Dunder-Mifflin. This is Pam. [listens] Uh, yeah. [snaps her fingers in the air, getting Jim's attention] Just one second. I will, uh, transfer you to our manager, Michael Scott.
Jim: Um... Whoa. [picks up ringing phone][in managerial voice] Michael Scott here. Yes, I am regional manager of this orifice. Mmm hmm. Dwight Schrute is amazing. Yeah. No, he is actually the single greatest employee of his generation. Mm hmm. You know what? I'm gonna tell you what. You hire Dwight K. Schrute, and if he does not meet, nay, exceed every one of your wildest expectations, well then, you can hold me, Michael Gary Scott, personally and financially responsible. Okay. Okay. Okay-kay-kay-kay-kay. Okay.
Dwight: Stanley, could you come with me, please.
Stanley: No.

Dwight: As Assistant Regional Manager...

Stanley: To the.

Dwight: Look! I've got some bad news. You're fired. You need to pack up your things and go. [Stanley laughs.] I'm serious, Stanley. It's over. I'm sorry.

Stanley: [laughs, and imitates Donald Trump] You're fired. Get your fingers off my phone.
Michael: So. How did it go with Stanley? How... how'd he take it?
Dwight: He wouldn't listen to me

Michael: Ahh, come on.

Dwight: If you want to fire him, you're going to have to tell him yourself.

Michael: I don't wanna fire Stanley. I never said that. I'm certainly not going to do it myself. Get those big, baleful, eyes staring at me. Yikes. Just, okay, just... [waves Dwight away]
Dwight: [whispering on the phone] Cumberland Mills?! And how did you get my resume? Oh no, no. I'm very flattered. Don't get me wrong. I'm just not sure that it's my official resume or if it's something that maybe a satisfied customer posted online. What does it say under martial arts training? Oh. Okay, I'm gonna have to supplement that. Could I have your fax number?
Dwight: Would I ever leave this company? Look, I'm all about loyalty. In fact, I feel like part of what I'm getting paid for here is my loyalty. But, if there were somewhere else that valued that loyalty more highly, I'm going wherever they value loyalty the most.
Oscar: Oh... hey.
Ryan: Oh, your dress is stuck in the back. Kind of just...

Oscar: Oh. [fixes his dress]
Dwight: [on the phone] So you got the fax? So why didn't you add it to the res... ? What do you mean? Of course martial arts training is relevant. Oh, excuse me! I know about a billion Asians that would beg to differ. Uh, yeah, I get a little frustrated when I'm dealing with incompetence. Well, you know what? You can go to hell, too. And I will see you there... burning. Fine! Okay... oh wait! So you'll let me know when you've made a decis... [stops and hangs up phone.]
Pam: Jim is really talented. And he should be the one who's getting a better job offer. Like, for real.
Pam: Don't take this the wrong way, but... you should go for that job.
Jim: Um... it's in Maryland.

Pam: Yeah, but I mean, look at the salary. And it's definitely a step up. And a challenge.

Jim: Yeah. Yeah. You know what? Maybe... maybe I will. [starts walking away]

Pam: Jim...
Dwight: This is called leveraging an offer. [walks into Michael's office] Michael, can I talk to you for a moment?
Michael: Oh, God.

Dwight: I just thought you should know that I was just offered a job with better pay, better benefits and a better title at Cumberland Mills.

Michael: Fantastic!

Dwight: And I turned it down.

Michael: What?! That would've solved all my problems.

Dwight: Out of loyalty to this company...

Michael: Oh, you idiot.

Dwight: ... so I was hoping to be made Assistant Regional Manager officially.

Michael: If you left, I wouldn't have to fire anybody.

Dwight: But then you wouldn't have me here.

Michael: Big deal. Oh, it would've worked out so well. Can you get it back?

Dwight: It's in Maryland.

Michael: You can call. Can you call 'em?

Dwight: I can't. I... I suppose I coul... no. They never really made me an offer anyway.

Michael: Wohahah! Why are you torturing me?! God.
Jim: Honestly, I don't think Michael has the slightest clue of who he's gonna fire. I think he keeps hoping that someone's going to volunteer. Uh, or be run over by a bus before the deadline. But in the end, really, what's going to happen is it's gonna be the first person to give him a dirty look in the hall. And therein lies the true essence of his charisma.
Michael: [clearing his throat and interrupting Jim's talking head] Can I speak to you a minute?

Jim: Um... yes.
Jim: Michael, I really didn't mean to...
Michael: Help. Me.

Jim: I'm sorry?

Michael: I want you to role play firing me. I want you to fire me, and I will take it.

Jim: Oh, you want me to be you?

Michael: Yes.

Jim: Okay.

Michael: I want you to be me, and I will be Creed.

Jim: Oh, are you firing Creed?

Michael: No, no, no. That's just the first thing... came... in head.

Jim: We should switch seats in order to...

Michael: Yes, that's a good idea.

Jim: Alright. [they stand up] Excuse me.[They sit down] I'm really sorry, but I have to let you go. And it's purely budgetary. It's not personal...

Michael: Aaaahh! I'm gonna k*ll myself!

Jim: Wow.

Michael: I'm going to k*ll myself, and it's your fault!

Jim: That's an overreaction.

Michael: Corporate is really breathing down my neck. And they're saying this has to be done by the end of the month.

Jim: Is this you? Are you being you, or is this Creed? Are you...

Michael: I... this is Creed.

Jim: Okay.

Michael: I'm improvising, so just try to keep up. [phone rings]

Jim: Oh, hold that thought. Hold that thought.

Michael: And I'm very angry, and I want...

Jim: [picks up the phone] Michael Scott here.

Michael: I'm gonna k*ll you. I'm going to k*ll you for firing me.

Jim: Toby? Mm hmm. [looks back to Michael] I really have to take this Creed, so it was really worth...

Michael: Get off, get off. No, no. OK.. just get off.[sits back down in his chair and waves Jim off.] Just, just... yeah.
Pam: What happened?
Jim: It wasn't me.

Pam: Oh. That was like crazy. 'Cause I was...

Jim: Yeah, I know.
Michael: Uh, hey... Creed?
Creed: Huh?

Michael: Could I talk to you for a second?
Michael: You are great. Very ambitious. And I feel like you want more than this little office has to offer. And I understand that you'd wanna just spread your wings, and fly the coop.
Creed: What are you telling me?

Michael: I... we're gonna have to... You... you want something better.

Creed: No, I don't. I wanna stay right here.

Michael: No, you wanna leave.

Creed: No, I wanna stay here.

Michael: Why... why are you making this so hard?

Creed: Um, I think there's a misunderstanding, Michael.

Michael: I think you're right.

Creed: Can I go?

Michael: No, of course you can't go. We haven't even started this horrible process of... okay, Creed. I need to let somebody go today. They told me I need to let somebody go. And as much as I think you're a great guy, and I like you, you're... you're, goodbye.

Creed: Let's f*ght it.

Michael: Hmm?

Creed: Let's call Jan and f*ght this thing together like the old days.

Michael: What old days? What are you talking about?

Creed: Did you start the paperwork yet?

Michael: It's right here on the desk, yeah.

Creed: You don't have to do this, Michael.

Michael: I can't, I can't...

Creed: Undo it!

Michael: I can't change anything. This is the way...

Creed: No, you have the power to undo it.

Michael: I don't... okay, just listen.

Creed: Michael, undo it!

Michael: Don't...
Michael: Yeah, I went hunting once. sh*t the deer in the leg, had to k*ll it with a shovel. Took about an hour. Why do you ask?
Michael: I have to fire someone today, okay?
Creed: Fine. Fire someone else. Fire Devon. He's terrible. I am so much better at my job than Devon.

Michael: Okay, well... I already picked you. And you know that. So, unless I just go through with this, you're always gonna look at me as the guy who almost fired you.

Creed: No, no, no, no, no, no. I will forget so fast. You will be my savior. You're they guy who gave me my life back. Thank you. I knew you'd see it my way Michael. God Bless you. You're a fine man.

Michael: Don't...

Creed: Listen, you will not regret this either. Devon is terrible; No one's gonna miss him. Good, good, good.
Michael: Devon, could I talk to you for a sec?
Devon: Creed's an idiot, you know that.
Michael: Well, he...

Devon: No, no, no, no, no, no! You had it right the first time.

Michael: Well, maybe I did.

Devon: Exactly. You gotta go with your gut, man.

Michael: Huh. No! I can't, no. I can't go back. I would look like an idiot.

Devon: That's why I'm being fired?

Michael: No.

Devon: So you might not look like an idiot?

Michael: No. It was all the stuff that I said. It was the business downturn, the cutbacks, and, and...

Devon: This is unbelievable!

Michael: I just hope that you and I can remain friends.
Michael: Devon, wait, please.
Devon: What!

Michael: Look, look. In addition to severance, and everything, I want to give you this gift certificate to Chili's. From me. Okay? No hard feelings.

Devon: [takes the gift certificate and tears it up] Kevin, Jim, Pam, Kelly, Toby, Oscar, Meredith, Phyllis, Stanley, or the temp. If any of you wanna meet me for a drink, I'm going to be at Poor Richard's. And the rest of you can go to hell!

Angela: [watching nearly everyone leave] What about the Halloween party?
Pam: Oh, hey, Jim. Wait, stop. Um, I'm sorry... for pushing you towards Cumberland. Seriously, if you left here, I would blow my brains out.
Jim: [motions for her to follow him] Come on.
Jim: That's just a figure of speech, you know? Blow your brains out? Come on. All it really means is that we're friends. Who else is she gonna talk to if I'm gone, right? I mean, if she left, I wouldn't blow my brains out. Of course, I would take that job in Maryland. Because it's double the pay, and soft shell crab just happens to be my favorite food.
Michael: I love Halloween. You know, it's just, it's just fun. Every year, it's just fun. Last Halloween I came as Janet Jackson's boob. It was topical. People got a... a big kick out of it. The year before that, I came as Monica Lewinsky, and I wore a stained dress. The year before that, I also came as Monica Lewinsky. And before that, I was O.J. It was pretty funny. Oh, I wish you were here last year.
Children: [ringing the doorbell of Michael's Condo] Trick or treat!
Michael: He... Hey, hey, hey, hey! How you doing? Wow! You guys looks great.

Kid: I'm a bumble bee.

Michael: You look great! And you're a princess?

Kid: A fairy princess.

Michael: A fairy princess. You're very... .

Kid: I'm a lion.

Michael: You're a lion. [trying to to open a bag of candy] Wow, I want to hear your, your... Oh! [the bag tears open, spilling all the candy] Oh, okay, that's all yours. That's all yours. Grab it, grab it. You know what? You guys are getting all of these.
Dwight: Where is my desk?
Jim: That is weird.

Dwight: This is not funny. This is totally unprofessional.

Jim: Ok, well, you're the one who lost the desk.

Dwight: I didn't lose my desk.

Jim: Okay, calm down. Where was the last place you saw it?

Dwight: Okay, who moved my desk?

Jim: I think you should retrace your steps.

Dwight: Ok, I am going to tell Michael and this entire office will be punished!

Jim: Colder... warmer... little warmer... there you go, ooh, warmer... warmer... warmer... warmer... warmer ... cold, cold, cold, back up... ooh, ooh, warmer, hot, red hot, hot, very hot.

Dwight: [In bathroom, answers phone] Dwight Schrute.

Jim: [On the phone with Dwight] Hi, Dwight, um, what sort of discounts are we giving on the 20 lb white model.

Dwight: Jim, I've given you this information, like, twenty times.

Jim: I know.

Dwight: It's by the ream?

Jim: Uh, yeah, ream.

Dwight: ...now, $9.78, signs and discounts 7%.

Jim: Ok, thank you, gotta get back to work.

Dwight: Wash your hands, Kevin.
Jim: [On the phone] Right, oh let me just check the pricing list. Hold on one second...
Dwight: [Also on the phone] Sensei, hello it's Sempai...

Jim: Umm...

Dwight: Dwight...

Jim: You know what, let me give you a call right back. I'm going to uh, find it and then I'll call you back, thanks.

Dwight: Yes, I just had a ques-... Yes Sensei. Arigatou gozaimashita. Hai.

Jim: Was that your mom?

Dwight: No, that was my Sensei.

Jim: Oh, I thought it was your mom.

Dwight: I am now Sempai, which is Assistant Sensei.

Jim: Assistant to the Sensei, that's pretty cool.

Dwight: Assistant Sensei.

Jim: Ok.
Dwight: I am a practitioner of Goju Ru Karate, here in Scranton. My Sensei, Ira, recently promoted me to purple belt, and gave me the duties of a Sempai. Not that a lot of people here in America know what a Sempai is, but it is equally as respected as a Sensei.
Stanley: I don't want to stay until seven again this year.
Pam: I don't really have any control over that Stanley.
Pam: Michael tends to procrastinate a bit whenever he has to do work. Umm, time cards, he has to sign these every Friday. Purchase orders have to be approved at the end of every month. And expense reports, all he has to do is initial these at the end of every quarter. But once a year, it all falls on the same Friday and that's today. I call it the Perfect Storm.
Michael: [singing and tapping on his coffee mug] I don't want to work, I just want to bang on this mug all day.
Ryan: Did you ask me here for any specific reason?

Michael: Uhh, yes, I did, here's the dizzle. I have a very top secret mission for you. I want you to update all the emergency contact information.

Ryan: Why is that secret? [Pam knocks and walks into Michael's office]

Michael: [to Pam] Hello, oh God, busy work. Ahh, get away, cretin.

Pam: Umm, I put stickers so you know where to sign.

Michael: Yes, thank you. I know where to sign.

Pam: It's just last year you...

Michael: Last year they were out of order, weren't they Pam?

Pam: Well, the last pick-up for overnight deliveries is at seven. So you need to have them signed by then. Or much earlier.

Michael: Chillax, Pam. Stop Pam-M-S-ing. That's pretty good. Um, actually, I'm sending Ryan on a top secret mission. Tell her what it is.

Ryan: Updating emergency contacts.

Pam: Well, is that really a priority?

Michael: Is it a priority? Oh I don't know, um, what if there is a tornado, Pam? People's legs are crushed under rubble. "Please, would you be so kind as to call my wife? No, I can't because we don't have any emergency contact information because Pam said it wasn't a priority." Think. Think with your head, Pam. Ok, well. She walks out. That's the problem with being a boss is that when you are tough they resent you and when you are cool they walk all over you.

Ryan: Catch-22.

Michael: Catch-22. Yes. Why don't you give me your contact information to start with, ok, what's your cell?
Jim: Uh, Larissa Halpert.
Ryan: What's her address? [Ryan's cell phone rings]

Jim: 117 Mount Bergin St.

Ryan: Hello?

Michael: [in his office on his cell phone, talking in a fake high voice] Hey Ryan. This is Michael Jackson calling from Wonderland.

Ryan: Do you mean Neverland?

Michael: This is Tito.

Ryan: What?

Michael: Calling from... [Ryan hangs up]
Pam: [Reading Jim's palm] You're major and minor lines cross at a ridge - that sucks.
Jim: You making this up as you go along, aren't you?

Pam: I am just following the website.

Jim: Well, at least I don't have cavities.

Pam: Yes, you have very nice teeth.

Jim: Thanks.
Ryan: Who is your emergency contact? [Ryan's phone rings]
Kevin: Stacy.

Ryan: [looks to see who is calling but doesn't pick up]

Michael: [Taps on the glass in his office to get Ryan's attention] Pick up.

Ryan: Hello?

Michael: [in a high pitched voice] This is Mike Tyson.
Jim: Hey, Dwight. As Sempai, do you think there is ever going to be a day where humans and robots can peacefully co-exist?
Dwight: Impossible. The way they're programmed... You're mocking me.

Jim: No I'm not.

Dwight: Look, I'm going to offer you a little piece of advice. I'm not afraid to make an example out of you.

Jim: Oh, that's not advice. What advice sounds like is this[/b]: umm, don't ever bring your purple belt to work because someone might steal it. [reveals Dwight's purple belt]

Dwight: Ok, give that back to me.

Jim: Ok, say please.

Dwight: No. That is not a toy.

Jim: Please?

Dwight: Please?

Jim: Good, and it absolutely is a toy. Arigatou.

Dwight: Arigatou. This is not a toy. This is a message to the entire office so they can see that I am capable of physically dominating them.
Michael: And this is more a ying-yang thing. The 'Michael' all cursive, the 'Scott' all caps. Left brain, right brain. Or, duality of man.
Pam: Could you practice on the forms?
Dwight: No women or children, unless provoked.
Jim: Ok, Roy?

Dwight: Warehouse guy. Doesn't count.

Jim: Ok. Michael? Could you b*at up Michael?

Michael: Yeah, yeah, I don't think that would happen.

Dwight: Because we're friends.

Michael: Because I would kick his ass.

Jim: Well, Dwight's a purple belt, so...

Michael: So? I've b*at up black belts.

Jim: Uh, how did you know they were black belts?

Michael: They told me. After. You see, I used to run with a very tough crowd. Street Fighter types. Real, real bad people, I'm just lucky I got out.
Ryan: Is your wife still your contact?
Toby: Um, ex-wife. Yeah. Um, her last name is 'Becker' now.

Ryan: 'Kay.

Toby: You don't need to write 'ex'.
Michael: And after that, nobody ever messed with the 'Damn Rascals' ever again.
Jim: Sounds tough. When you're a Jet, [starts snapping] you're a Jet all the way, right?

Michael: You were a Jet?

Angela: Have you signed the expense reports yet?

Michael: Yes, in theory, I have. I just need to cross some t's and dot some i's. Alright, I'm going to be in my office if anybody needs me. [Puts Dwight in a headlock] Hoo-hah. Oh, wow, sleeper hold. That's my rebuttal. Shhh. Hoo. You are the weakest link.

Dwight: Argggg!
Michael: I'm friends with everybody in this office. We're all best friends - I love everybody here. But sometimes your best friends start coming into work late, and start having dentist appointments that aren't dentist appointments. And that's when it is nice to let them know that you can b*at them up.
Michael: Just hit me. You'll see.
Jim: I can't. I just got a manicure.

Michael: Oh, q*eer... [realizes he is on camera] eye. q*eer eye. That's a good show. Important show. Go ahead. Do it.

Jim: Just have Dwight punch you.

Michael: Oh yeah, that would be kind of worthless because I know a ton of fourteen year old girls who can kick his ass.

Jim: You know a ton of fourteen year old girls?

Dwight: What belt are they?

Michael: Look, Dwight is a wuss. When we rented 'Armageddon'...

Dwight: No!

Michael: ... he cried at the end of it. He did.

Dwight: Michael, I told you, it was because it was New Year's Eve and it began to snow at exactly midnight.

Michael: Oh, Bruce Willis. Are they going to leave him on the asteroid?

Dwight: Ok, I'll punch you.

Michael: Ok, here we go. Alright, come on.

Dwight: Kiyah!

Michael: Fuuuaaaahhhhh... oohhhhh!
Dwight: Did I want to harm Michael? The one man I've been hired to protect? No, I did not.
Jim: Are you ok? Are you sure you are alright?
Michael: Yeah. [Jim opens office door for Michael] Thank you.
Dwight: I come from a long line of fighters. My maternal grandfather was the toughest guy I ever knew. World w*r II veteran. k*ll 20 men then spent the rest of the w*r in an Allied Prison Camp. My father battled blood pressure and obesity all his life. Different kind of f*ght.
Jim: Ok, he has to be stopped. Please, please, please, please, just ask Michael.
Pam: I don't know.

Jim: Ok, I'll buy you a bag of chips.

Pam: French Onion?

Jim: Obviously.

Pam: Ok.

Jim: Yes.

Dwight: [to Kevin, who he is teaching to f*ght] Take this pen and s*ab me with it.

Michael: [Pam knocks on his door] Go away.

Pam: I just have a quick question.

Michael: I haven't signed them, ok?

Pam: No, it's not that. Um, I was just wondering, since I'm probably going to have to stay late, could you ask Dwight to stay late too so he can walk me to my car?

Michael: Come in. Um, Pam, I hate to break this to you but Dwight can't stop you from being mugged. He's just not tough enough.

Pam: He's a purple belt. That's really high.

Michael: Oh, I could b*at up Dwight. That's ridiculous. I could m*rder him.

Pam: It's just out there, you...

Michael: Oh, so that's what they are saying?

Pam: Yeah.

Michael: Ok, alright, where is Dwight?

Jim: Uh, Kitchen.

Michael: Ok.

Kelly: Hi-yah!

Dwight: Good.

Kelly: Wow, that's actually pretty cool Dwight.

Dwight: Now watch, let me take you from behind.

Kelly: What?

Michael: Watch out Kelly, he might sucker punch you.

Dwight: I didn't sucker punch you, Michael.

Michael: No, Really?

Dwight: In case you remember, I was defending my honor... like a samurai.

Michael: Really? Well, the offer, Dwight, was for one punch which I absorbed. I had no idea that there would be a second punch. So, catch-22.

Dwight: Ok, fine. Tit for tit. Give it your best sh*t. Two punches. Go!

Michael: Look, if we were in a bar right now, there would be two punches[/b]: me punching you and you hitting the floor.

Dwight: No, I would block your first punch rendering it ineffective.

Michael: Really?

Dwight: Yeah.

Michael: You know what? You're just lucky that we are at work right now.

Jim: Ooh, what about, uh, Dwight's dojo?

Michael: No, they must have class.

Dwight: No, it's free during the day. It's fine.

Michael: Look...

Dwight: I've got the key.

Toby: Michael...

Michael: Hey, Toby.

Toby: Any word on those time cards?

Michael: I've got an idea[/b]: why don't you leave right now. Why don't you walk away from the room, 'kay? Fine. We'll go at lunch. Pam, make an announcement. Figure out carpools.
Jim: Um, well, we are all getting excited to see this f*ght. The Albany branch is working right through lunch to prevent downsizing, but Michael, he decided to extend our lunch by an hour so we could all go down to the dojo and watch him f*ght Dwight. f*ght... f*ght, f*ght, f*ght, f*ght, f*ght, f*ght, I'm coming, f*ght...
Michael: I recognize that. That is Japanese for California Roll.
Ira: Uh, no, it's not.

Michael: I think it is. A guy told me about that.

Ira: Actually, it's a symbol for eternal discipline.

Michael: Oh.
Jim: [Reading Pam's palm, while she has on extremely padded gloves] Wow, that is really interesting.
Pam: What?

Jim: Your love line- I'm just kidding. I can't see anything.

Pam: Well, look closer.

Jim: [Jim moves his head closer and Pam taps him gently in the face] Oh, ok.

Pam: Once point for me.

Jim: [Gently taps Pam on the forehead] Tied up.

Pam: Oh, you're dead.

Jim: What, what are you going to do? Bring it, Beesley. Bring it. Oh yeah, good move. Not such an ultimate fighter now.

Pam: Hey, put me down. Put me down. [Meredith turns and looks at Jim and Pam] Oh my god, hey, put me down. Hey...
Ira: Ok, gentlemen, listen up. After a clean strike to the chest, stomach, or kidneys, I will separate you and award a point. The first person to three wins. Alright?
Dwight: Yes, Sensei!

Michael: Alotta rules. Alotta rules. On the street we didn't have any rules. Maybe one - no kicks to the groin, home for dinner.

Ira: Shi mate!

Dwight: Hiii! [kicks Michael]

Michael: Hey!

Ira: Alright, break.

Michael: What the hell was that?

Dwight: Yes!

Ira: Dwight - awarded a point.

Michael: No.

Dwight: Eat it!

Michael: Alright, that's the way you want it.

Dwight: Two more.

Michael: Play dirty, huh? Ok, game on, man.

Kevin: Sweep the leg.

Michael: I'm comin' atcha man. Ok, purple belt, ok. I got him.

Dwight: No.

Michael: I got his pants.

Dwight: It was my pants.

Ira: No points for pants.

Michael: Dwight, you have... No, you have something... God, you look like such an idiot! [Lots of yelling and flailing of arms by Michael and Dwight]

Ira: Clean single kick, gentlemen.

Michael: Go on, I dare you to kick there again. Kick there again, I dare ya.

Ira: Ok, break. Break.

Dwight: No holding.

Michael: You can't see. You can't see. Good boy. Good boy. Great boy. Two points, three points, four points. I win. I win. [Michael is using his head guard to hit Dwight] Eight points. Nine points. [Begins to hock a loogie]

Dwight: No, stop it! Come on! Michael.

Michael: Open your mouth.

Dwight: No, Michael!
Michael: You talkin' to me? You talkin' to me? "Raging Bull." Pacino. Oh, I want that footage. I want it. I need it. Ah, I have to get back to work. I have lots of work... Oh, oh check this out. Come here. [Michael opens his blinds and looks at Ryan in the parking lot] There he is. Mr. Temp. Having lunch by the car. Let us play with him. This'll be hilarious. [Calls Ryan on the phone, Ryan doesn't pick up after seeing that Michael is calling] Oh, we're playing phone tag.
Ryan's Voicemail: Seven new messages. First New Message. [Michael's voice] "Hi, Ryan. This is Saddam Hussein." Next new message. "Hi, Ryan. This is your girlfriend... and I'm mad!"
Michael: My emergency contact is Todd Packer. Todd F. Packer. Do you know what the F. stands for?
Ryan: Fudge?

Michael: [knock at the door] Yeah... uh, come in. Oh, hey Karate Kid. The Hillary Swank version. Hi. How are ya?

Dwight: I need to change my emergency contact information from Michael Scott.

Ryan: Ok, to what?

Dwight: Just put "The Hospital." Contact number[/b]: just put 9-1-1. [Dwight leaves]

Michael: He is such a sore loser. You heard, obviously, that I mopped the floor with him this afternoon. You know what, um, do yourself a favor and just keep me as his contact and I will call the hospital. Cut out the middle man.
Kevin: Later Jim.
Jim: Later, Kev. [Puts French Onion Potato Chips on Pam's desk] Have a good weekend.

Pam: Yeah, you too.
Michael: [Knock at the door] Yeah.
Ryan: I have the emergency contacts.

Michael: Yeah, just throw them on the chair. I'll take it from here. So, whatcha up to this weekend?

Ryan: Uh, hanging out with some friends, probably.

Michael: If you're doing anything crazy, give me a shout.

Ryan: Yeah, alright, I'll um, see you Monday.

Michael: Alright, bye.
Michael: Dwight?
Angela: Michael, did you finish yet?

Michael: This close. Dwight, may I speak with you for a minute?

Dwight: I'm busy.

Michael: Well, [points at himself] busier. Making the time.

Stanley: Michael, can't your conversation wait till Monday.

Toby: We want to go home.

Michael: Well, you don't even have anyone to go home to, Toby.

Pam: The shipping place closes in a half hour.

Michael: I know, but I've been carrying the load on my back all day, and if everybody would just chip in a little bit, it'd might help me out. What do you say? Let's g*ngb*ng this thing and go home. Good? Dwight.
Angela: This is illegal.
Stanley: I don't care.
Michael: I have been testing you the entire day. Did you know that?
Dwight: Of course.

Michael: And I am happy to say that you have passed. So effective immediately I am promoting you from Assistant to the Regional Manager to Assistant Regional Manager.

Dwight: Michael, I don't know...

Michael: I know, I know, I know, I wouldn't be offering it if I didn't think you could handle it.

Dwight: I can handle it. I can. Wow. So I guess this will just be my office.

Michael: No, no, title change only.

Dwight: I'll have Pam send out a memo.

Michael: No, no. Three month probationary period. Let's not tell anybody about this right now.

Dwight: Just a formality.

Michael: Absolutely but not really.

Dwight: Michael, I have so much to learn from you.

Michael: Yes you do.

Dwight: Thank you, Sensei.

Michael: And, ditto.
Michael: I told Dwight that there is honor in losing. Which, as we all know, is completely ridiculous, but there is, however, honor in making a loser feel better which is what I just did for Dwight. Would I rather be feared or loved? Um, easy. Both. I want people to be afraid of how much they love me. And I think I proved that today at the dojo.
Ryan: [entering office] Hey, have they left for the big meeting yet? I've got Michael's lucky tie.
Jim: No. They're in the conference room.

Ryan: Good.

Pam: Wait, are those Michael's Levis?

Ryan: Yeah, who dry-cleans jeans?
Pam: Michael and his jeans. He gets in them, and I'm not exactly sure what happens. But I can tell you, he loves the way he looks in those jeans. I know that's why he started casual Fridays.
Pam: [to Ryan] I'll take those. Thanks. [throws jeans under her desk]
Jan: This is a projection of the county's needs...
Michael: Wow, graphs and charts, somebody's really been doing their homework. Looks like USA Today.

Jan: Thirteen schools, uh, two hospitals...
Jim: So this possible client they're talking about, actually a big deal. It's Lackawanna County. Our whole county. And if we get this, they may not have to downsize our branch. And I could work here for years. And years. [groan] Years.
Jan: So when we get to the Radisson, I'd like to, um-
Michael: I changed it. To Chili's.

Jan: Excuse me?

Michael: Radisson just gives out this vibe, "Oh, I'm doing business at the Radisson". It's kind of snooty. So.

Jan: You had no right to do that, Michael.

Michael: Here's the thing. Chili's is the new golf course. It's where business happens. Small Businessman Magazine.

Jan: It said that.

Michael: It will. I sent it in. Letter to the editor.

Jan: Alright. But you will let me run this meeting.

Michael: Uh huh, uh huh. [under his breath] Power trip.

Jan: What?
Oscar: She had done a background check on me, she had it printed out.
Jim: No...

Oscar: Yeah. And she was asking me about stuff, line by line, while we were having dinner.

Toby: That is unbelievable.

Pam: What is going on?

Jim: We are doing worst first dates.

Pam: Oh my God, I win! Ok, it was a minor league hockey game. He brought his brother, and when I went to the bathroom, the game ended and they forgot about me.

Oscar: Ok, that's a joke.

Pam: No, they had to come back for me.

Jim: Wait, when was this?

Pam: Umm... it was not that long ago.

Kelly: Wait, not Roy. Say it's not your fiance. [laughs]
Jim: I always knew Pam has refused to go to sports games with Roy, but I never knew why. Interesting.
Michael: Ok, let's do this thing. [to Pam] Wish us luck.
Dwight: Good luck, Michael. Good luck, Jan.

Jan: Thank you.

Michael: [under his breath] Kiss ass. Ok, probably going to go late tonight. Burning the midnight tequila. So, I think you could all just take off now.

Jan: Michael, shouldn't take more than an hour.

Michael: Well...

Jan: Do you always shut down the entire office when you leave for an hour?

Michael: No, no. That would not be efficient. Actually, they just don't get very much work done when I'm not here. [Jan stares at Michael] That's not true. I know how to delegate, and they do more work done when I'm not here. Not more. The same amount of work is done, whether I am here or not. [another Jan stare] Hey, everybody, listen up. This is what we're gonna do. You sit tight, until I return. Sound good? Doesn't matter, it's an order. Follow it blindly, mwahahaha, ok? Alright, ciao. [to Oscar] Adios!
Jan: So which way is Chili's?
Michael: Uh, I'll drive.

Jan: Oh, no, that's alright. I wanna leave straight from there.

Michael: It's just a couple blocks away, so... boy, you really don't know Scranton, do you?

Jan: I know Scranton.

Michael: At all!

Jan: Alright.

Michael: You ever been to Scranton, Jan? Dar de-

Jan: If it's a couple blocks away-

Michael: Dar de dar.

Jan: Ok.
Michael: Jan Levinson-Gould. Jan is cold. If she was sitting across from you on a train and she wasn't moving, you might think she was dead.
Michael: We should come up with a signal of some sort.
Jan: Why would we need a signal?

Michael: Well, in case one of us gets into trouble, the other one can signal-

Jan: What kinda trouble are you planning on getting in, Michael?

Michael: Well, I... it could be either of us.

Jan: You're gonna let me do the talking, we agreed on that.

Michael: Yeees.
Michael: Hello? Christian?
Christian: Yes.

Michael: Thought that was you. Hi. Michael Scott. This is Jan Levinson-Gould.

Jan: Just Jan Levinson.

Michael: No Gould?

Jan: No. [To Christian] Thank you very much for meeting with us. Have you been waiting long?

Christian: No, not long.

Michael: Uh, Jan, what happened?

Jan: Michael.

Michael: Is Gould dead? What uh-

Jan: Michael, we got divorced, ok? [to Christian] I'm so sorry. Excuse me.

Michael: Wow, you're kidding me! Do you wanna talk about?

Jan: Michael. [to hostess] Uh, could we have a table for three, please?

Michael: When did this happen?

Jan: We're in a meeting.

Michael: Ok.

Hostess: This way, please.

Jan: Christian.

Michael: Alright, after you.

Christian: Thank you.

Michael: [mouths "Wow" to the camera]
Jan: I thought we could start by going over the needs of the county.
Christian: Right. Well, Lackawanna County has not been immune to the slow economic growth over the past five years. So for us, the name of the game is budget reduction-

Michael: Awesome blossom.

Jan: What?

Michael: [to Christian] I think we should share an Awesome Blossom, what do you say? They are awesome. Want to, Christian, blossom?

Christian: Sure.

Michael: Ok, it's done. Actually, [turns around] Megan, may we have an Awesome Blossom, please, extra awesome? Now it's done.

Jan: So-

Michael: I heard a-

Jan: If you have a-

Michael: Very very funny joke the other day. Wanna hear it?

Jan: Christian, you don't have to listen to this.

Christian: It's ok, I like jokes.

Michael: Ok.

Jan: Just the one.

Michael: Just one joke. Ok. Well, if it's just gonna be one, I will think of a different joke. Umm... let's see... choo choo choo.
Pam: Dunder-Mifflin, this is Pam.
Michael: Pam, it's Michael. I need you to go into my office and check some data for me.
Pam: [to Michael on speakerphone] Ok, you want me to read 'em?
Michael: Yes.

Pam: Ok. Um, a fisherman is walking down Fifth Avenue walking an animal behind him-

Michael: No.

Pam: When-

Michael: Nope. Told it. Not as good as you think. Pick another one.

Pam: Ok. There's a transcript between a naval ship-

Michael: Oh ho ho, yea! Bingo! And a lighthouse. Yes. That is hysterical. Could you start that one from the beginning?

Pam: Sure. There's a transcript between a naval ship and a lighthouse.
Jim: Is this real? [Pam dumps Michael's screenplay on Jim's desk]
Pam: It is a screenplay. Starring himself.

Jim: Agent Michael Scarn.

Pam: Of the FBI.

Jim: How long is this? [flips through pages] Oh, Pam. Good work! Oop, wait, stop. Drawings.

Pam: What is that?

Jim: Oh, those are drawings. In case the writing didn't really put a picture in your head. And there he is, in the flesh, Agent Michael Scarn. Now we know what he looks like.
Michael: First guy says "Well, I'm an astronaut, so I drive a Saturn". And the second guy says, "Well, I am a pimp, so I drive a cheap Escort". And the third guy says "I gotcha both b*at, I'm a proctologist, so I drive a brown Probe".
Christian: Ohhh no! [laughs] Oh my God, that's funny! I almost had Awesome Blossom coming out of my nose!

Jan: [to waitress] Excuse me, could I have a vodka tonic, please?
Jim: Do we all have our copy of "thr*at Level: Midnight", by Michael Scott?
Everyone: Yeah, yeah.

Jim: Alright, let's get this started. I'm gonna be reading the action descriptions, and Phyllis, I would like you to play Catherine Zeta Jones.

Phyllis: That's the character's name?

Jim: Oh yeah-

Dwight: Ok, you guys should not be doing this.

Jim: Why not, Dwight? This is a movie. I mean, this is for all of America to enjoy.

Dwight: You took something that doesn't belong to you.

Jim: Dwight-

Dwight: Brought it in here-

Jim: Do you want to play-

Dwight: Made copies of it-

Jim: The lead role of Agent Michael Scarn?
Michael: [making the mouth on his tie talk] Yum! Yum yum yum! [Christian laughs] That's delicious! I love it!
Jan: We would probably be upset with ourselves if we went this whole night without talking business, so, Dunder-Mifflin can provide a level of personal service to the county that the warehouse chains just can't match.

Christian: Well, we are out to save money.

Jan: What's the bottom line?

Michael: Bop! Be blah bop, be boo boo bop.
Michael: That's why I wanted a signal, between us, so that I wouldn't have to just shout non-sense words. That's her fault.
Michael: Did somebody say "baby back ribs"? Hmmm? Hmmmmm?
Jan: I don't think Christian has time for that.

Christian: I have time.

Michael: [singing] I want my baby back, baby back, baby back [Christian laughs]

Michael and Christian: [singing] I want my baby back, baby back, baby back-

Michael: [singing] Chili's baby back ribs...
Jim: [reading the screenplay] Inside the FBI, Agent Michael Scarn sits with his feet up on his desk. Catherine Zeta Jones enters.
Phyllis: Sir, you have some messages.

Dwight: Not now!

Phyllis: They're important.

Dwight: Ok, what are they?

Phyllis: First message is[/b]: "I love you". That's from me.

Dwight: Not in a thousand years, Catherine. We work together. And get off my desk!
Dwight: Yes, I have acted before. I was in a production of "Oklahoma" in the seventh grade. I played the part of Mutey the Mailman. They had too many kids, so they made up roles like that. I was good.
Dwight: If it isn't my old partner, Samuel L. Chang.
Ryan: Agent Michael Scarn, you lost some weight.

Dwight: Thank you for noticing. Now keep me company for one more mission. [Pam gets up to talk to Roy]

Pam: Hey, uh, I have to work late.

Roy: [looks around conference room] You're joking right?

Jim: Michael Scarn takes out a nine-millimeter g*n and sh**t the-

Dwight: Pow! Pow! Pow!

Ryan: Hahaha, Agent Michael Scarn, you so funny. Word.
Kevin: Michael's movie? Two thumbs down. [Smiles] Heh.
Jim: A man sitting several seats down, who has a gold face, turns to Michael Scarn. [out of character] Uh... Ooh, Oscar, you wanna play Goldenface?
Oscar: Mr. Scarn, perhaps you would be more comfortable in my private jet?

Dwight: Yes, perhaps I would, Goldenface. Sam, get my luggage.

Ryan: I forget it, brutha.

Dwight: Samuel, you are such an idiot, you are the worst assistant ever. And you're disgusting, Dwigt. [out of character] Wait, who's Dwigt?
Pam: Here's what we think happened. Michael's sidekick, who all through the movie is this complete idiot who's causing the downfall of the United States, was originally named Dwight. But then Michael changed it to Samuel L. Chang using a search and replace, but that doesn't work on misspelled words, leaving behind one Dwigt. And Dwight figured it out. Oops.
Dwight: D-W-I-G-H-T.
Dwight: Ok, you know what? I am done with this. That's it, the end.
Jim: Well, some of us wanna keep reading, so-

Dwight: Uh, you don't speak for everyone, Jim. Ok, announcement. My uncle bought me some fireworks. Anyone who wants to see a real show, come with me outside now.

Jim: That's actually a good idea. We'll all take a brief intermission. [To Pam] Hey, are you hungry?

Pam: Yeah.

Jim: Yeah?
Christian: So after watching my mom go through so much pain, I decided to keep that promise, that I made to her, and take care of her.
Michael: Woo, well, this brings us to Jan. Truth or Dare? Tell us about your divorce. Ohh, ohh.

Jan: Oh no, Michael, Michael, please. No, really.

Michael: Oh, so you're not gonna play? She's not playing.

Christian: It's not fair.

Michael: She's not playing the game.

Jan: We'd been fighting for a while-

Michael: Check please.

Jan: He didn't want kids, but I knew that going into it. But he also knew that I did. I guess I thought that he would change his mind; he thought that I would change mine.

Christian: You didn't.

Jan: I was stupid.

Michael and Christian: No.

Michael: No, you were not stupid. Gould was stupid. Right?

Christian: That's right.

Michael: You know?

Christian: You were really brave! You, you put your arms out there, you slit your wrists.

Michael: It's true.

Christian: You said "World, this is my blood! It's red, just like yours. So love me!"
Jim: I had plans to meet a friend tonight. Which I had to cancel. But this is cool, too. I'm not a complainer.
Jim: [Pam lights a candle] Wow.
Pam: For the bugs.

Jim: Nice. That's excellent, because bugs love my famous grilled cheese sandwich.

Pam: Yes... nice! I can't remember the last time someone made me dinner.
Christian: Right down the street?
Michael: Uh huh, Kenneth Road, born and raised. Spent my whole life right here in Lackawanna County and I do not intend on movin'. I know this place. I know how many hospitals we have, I know how many schools we have. It's home, you know? I know the challenges this county's up against. Here's the thing about those discount suppliers. They don't care. They come in, they undercut everything, and they run us out of business, and then, once we're all gone, they jack up the prices.

Christian: I know.

Michael: It's bad.

Christian: It's terrible.

Michael: It, you know what, it really is.

Jan: Uh- [Michael signals for her to shh]

Christian: I don't know. I guess I could give you guys our business, but you have to meet me half way, ok, because they're expecting me to make cuts.

Michael: Well, corporate's gonna go ballistic, but, uh, you think we could Jan?
Jim: So, I guess I'll see you in [looks at watch] ten hours.
Pam: What are you going to do with your time off?

Jim: Travel. I've been looking forward to it. It's gonna be... really nice. Gonna find myself.

Pam: [points to Jim's iPod] You have new music?

Jim: Yeah. [Pam puts her hand out for an earbud] Definitely.
Michael: [waving to Christian] See ya.
Jan: Bye... thanks. [pumps fist] Yes!

Michael: We did it!

Jan: We got it!

Michael: Nailed it. Nailed it! Come here.

Jan: I am really- [Michael kisses Jan] Thrilled. [Michael and Jan kiss again] Let's go.

Michael: What!?

Jan: Let's go.

Michael: Goin'. Ok. Where we goin'? Doesn't matter. Goin' to the go go. [nervous laugh] Oh-ok.
Dwight: [waking up on office couch] Michael? Michael? [goes into Michael's office] Michael? [looks out Michael's window] His car's not in the parking lot. I should check the accident reports. [taxi pulls into Dunder-Mifflin parking lot] Who's this? Jan?
Michael: Morning, Pam. Hey.
Michael: No, nothing happened. I-I swear, nothing happened. What, I'm, totally being serious. A gentleman does not kiss and tell, and neither do I. [laughs] No, seriously, guys, I'm not, I don't want to go into it at all. It's off limits. Fine, I took her back to her hotel and we made out for a little while. It was great. I mean she told me about her divorce, we talked for about five hours, she fell asleep on my arm. So.
Michael: Hello, Dwight.
Dwight: Did you do her?

Michael: Who.

Dwight: Jan Levinson-Gould.

Michael: Uh, no, no, no Gould.

Dwight: Did you do her?

Michael: This is none of your affair because she is your boss-

Dwight: And she is your boss.

Michael: And she is a woman. She is a strong, soft, thoughtful, sexy woman. And you know what? I don't think that I can sit here and let you talk about her that way without me defending her honor. [to camera] Jan, I defend your honor. [to Dwight] Is that all?
Jim: Jan didn't come back for her car last night.
Pam: What!?

Jim: Could it be that Agent Michael Scarn has finally found his Catherine Zeta?

Pam: Oh, I don't know... [Jim laughs, phone rings] Oh my God. This is Jan's cell.

Jim: No way.

Pam: Dunder-Mifflin, this is Pam.
Michael: I know we have to register as a consensual sexual relationship with HR. My question[/b]: do I do it as the man? Does she do it as my superior? I don't know. That leads to other issues that we may have in our relationship. It's, uh, [phone rings] Excuse me. Hello? Hi! Just talking about you. The camera? No. Uh huh. How's traffic? I miss you. What. Ok. Well, if it was a mistake, it was a wonderful mistake. No. [to camera] Would you excuse me? [to Jan] No, I did not intentionally get you drunk. Um hmm. No, no. [goes under his desk] This is just a f*ght. This is just a first f*ght of many fights we're gonna have. Right. No. Wha-so-I don't understand, you wanna see other people. Only other people. Wh-why, ok, I think you're still a little bit drunk [to camera which is now under desk] Excuse me? Excuse me?! [to Jan] I think you're, yes, why don't you just come back here, go to the hotel, have a few drinks and-no, no. I didn't slip you something!
Jim: Some might even say that we had our first date last night.
Pam: Oh, really?

Jim: Really.

Pam: Why might some say that?

Jim: Cause there was dinner, by candlelight.

Pam: Uh hmm.

Jim: Dinner and a show, if you include Michael's movie. [Pam nods reluctantly] And there was dancing and fireworks. Pretty good date.

Pam: We didn't dance.

Jim: You're right, we didn't dance. It was more like, swaying. But still romantic.

Pam: Swaying isn't dancing.

Jim: Least I didn't leave you at a high school hockey game.

Pam: I have some faxes to get out.

Jim: Oh, come on, Pam. I-
Jim: Ok, we didn't dance. I was totally joking anyway. I mean, it's not really a date if the girl goes home to her fiance. Right?
Dwight: [bouncing on an exercise ball] You should get one of these.
Jim: No. Thank you.

Dwight: Do you even know what this is? It is a fitness orb and it has completely changed my life. Forget everything you thought you knew about ab workouts.

Jim: Done.

Dwight: This ab workout is specifically designed to strengthen your core. [knocks things around Jim's desk] Sorry.

Jim: S'ok.

Dwight: Numerous health benefits, strengthens your back, better performance in sports, more enjoyable sex.

Jim: You're not having sex.

Dwight: Plus, improves your reflexes [knocks over more stuff] see, I would have caught that.

Jim: Ok, you know what, uh, how much is that?

Dwight: It's only twenty-five bucks.

Jim: Wow. Um, ok. [pops Dwight's orb with scissors]
Michael: Pam, could I see you in my office?
Pam: It's performance review day, company-wide. Last year, my performance review started with Michael asking me what my hopes and dreams were, and it ended with him telling me he could bench-press 190 pounds. So, I don't really know what to expect.
Michael: Pam, you're trustworthy-
Pam: Thank you.

Michael: And a woman-

Pam: Oh, no.

Michael: And I want you to listen to a voicemail from my boss. [Jan on recording] "Michael, it's Jan. I guess I missed you. I'll, uh, be there this afternoon for performance reviews. I hope it's understood that that will be our only topic of discussion. See you soon." First impressions?

Pam: Uh, just off the top... I think she'll be here this afternoon.
Michael: My boss is coming in today, the lovely Jan Levinson-Gould will, well, no Gould. The Gould has been [makes slashing neck hand motion] swack, divorced. Um, the awkward part is that this will be the first time that we'll be seeing each other since, well, uh, it was really nothing. We just sort of got caught up in the moment. The vulnerable divorcee gives herself to the understanding, with rugged good-looks, office manager. Just, uh, she didn't want it to continue for some reason. It, we both, I didn't want it, we both didn't want it to continue. Was not professional. Um, when people say something's mutual, it never is. But this was mutual.
Michael: [playing Jan's message] "I guess I missed you." I guess I missed you. So, she misses me?
Pam: She missed you.

Michael: But then she goes on to say "that will be our only topic of discussion". That doesn't mean anything, those are just words.

Pam: I have one idea of what it means.

Michael: Ok. Yeah, what, what?

Pam: Well I don't think you're gonna be very happy with this.

Michael: Ohhh, great. Well, now I'm in a terrible mood. Let's do your performance review-

Pam: Because she's conflicted. She has to be professional, but she's fighting feelings... for you.

Michael: Ah, why, that's great news? That, that, then why would, why would I not like that?

Pam: Um, just cause, that, you work together, and it might be awkward.

Michael: Oh, wow, wow. Alright, let's listen to that again. [plays Jan's message] "Michael, it's Jan. I guess I missed you".
Dwight: Oh, hey, listen, Jim. Here's a little tip for your performance review.
Jim: Ok.

Dwight: Tell Michael that we should be stocking more of the double-tabbed manila file folders.

Jim: We don't have double-tabbed manila file folders.

Dwight: Oh, yes, we do.

Jim: No, we don't.

Dwight: Yeah, it's a new product. So, you should just suggest that to him and he'll be sure to give you a raise.

Jim: Alright... well, I'm not asking for a raise. I'm gonna actually be asking for a pay decrease.

Dwight: Uh, that is so stupid. What if he gives it to you?

Jim: I win.

Dwight: Ugh, you know what? I am going to zone you out for the rest of today. I need to stay focused, and I don't have to see you tomorrow or Sunday and please don't call me, and we'll see how things go on Monday. Uh, stupid.

Jim: Wait, wait; one thing. Uh, by tomorrow, you mean Saturday, right?

Dwight: Uh, duh.

Jim: Duh.
Jim: Today is Thursday. But Dwight thinks that it's Friday. And that's what I'll be working on this afternoon.
Stanley: Sometimes women say more in their pauses than they say in their words.
Michael: Really?

Stanley: Oh, yes. Let's listen to it again. And this time, really listen to the pauses.

Michael: God, Stanley, that's frickin' brilliant. How do you know that? Did you learn that on the streets? Sorry.

Stanley: Oh, no, that's ok. I did learn it on the streets. On the ghetto, in fact.

Michael: No kidding.
Stanley: It's all about my bonus.
Pam: Michael and Jan definitely made out.
Jim: Ohh...

Pam: Maybe more.

Jim: Eck!... Oh! Also, it is Thursday, but Dwight thinks it's Friday. So, keep that goin'.

Pam: Oh, yea!
Michael: Good work, Stanley. Great performance review. Stanley in the house, everybody. Woo! Angela, your turn.
Angela: I actually look forward to performance reviews. I did the youth beauty pageant circuit. And I enjoyed that quite a bit. I really enjoy being judged. I believe I hold up very well to even severe scrutiny.
Pam: Michael?
Michael: Yeah?

Pam: Jan's on the phone for you.

Michael: Oh; Angela, you were totally satisfactory this year.
Michael: Interesting. Jan is calling me. Maybe it wasn't so mutual after all. [puts Jan on speakerphone] Yeah?
Jan: Michael.

Michael: Jan! To what do I owe this pleasure?

Jan: I am returning your many calls.

Michael: Well, hello to you, too. Um, yeah, I was just um, I just wanted to get some closure on uh, what transpired between us at the meeting we had in the parking lot of the Chili's.

Jan: No. No, we won't be discussing that, Michael. The only things I wanna talk about during your performance review are your concrete ideas to improve your branch.

Michael: Well, surely this uh, review is a formality because of what happened uh, at our meeting in the parking lot of Chili's.

Jan: Uh, your review is anything but a formality, Michael.

Michael: Oh.

Jan: I expect you to forget anything that you think may have happened between us and exhibit completely professional behavior.

Michael: Been thinking about you.

Jan: Ok, that is an example of completely unprofessional behavior.

Michael: Um, I don't see how that's unprofessional. Just-

Jan: Michael.

Michael: Yep.

Jan: Are the cameras with you...

Michael: No.

Jan: ...in your office?

Michael: They are not. Yes, they are. [Jan hangs up] That's my girlfriend.
Kevin: I heard they made out and had sex.
Oscar: No, they just made out. That's it.

Kevin: Well, I heard they made out and had sex.

Angela: Don't talk about it. Office romances are nobody's business but the people involved.

Kevin: Romances?
Michael: Pam, I have ideas on a daily basis. I know I do. I have a clear memory of telling people my ideas. Um, is there any chance you wrote any of my ideas down? In a folder? A "Michael-idea" folder?
Pam: Sorry.

Michael: That's unfortunate. How 'bout the suggestion box? There's tons of ideas in there.

Pam: What suggestion box?

Michael: The suggestion box that I put out, and people could be put in suggestions anonymously? Maybe there's prizes?

Pam: Oh, yeah. Uh, I think I remember that from back from when I first started.

Michael: Why don't you find it and tell people to get theirs... never mind, I'll tell them. Hello, everybody? Yeah, uh, attention, please. Jan Levinson's coming, very soon, and so, we're going to have our weekly suggestion box meeting, so you can all get your constructive compliments in a.s.a.p.

Ryan: Don't you mean constructive criticism?

Michael: What did I say?

Kelly: You said "constructive complements"; that doesn't make any sense.

Michael: Well, Kelly, that was neither constructive nor a compliment, so maybe you should stop criticizing my English and start making some suggestions. 'K?
Jim: [on phone] Hey, Dan, this is Jim, and it is about 11:15, and I wanted to know what you were up to tomorrow, which is the fifteenth, and that is a...
Dwight: Saturday.

Jim: [pumps fist] ...Saturday, so just let me know what you're doing tomorrow, Saturday, for lunch. Ok, talk to you soon.
Jan: [on phone] We'll address this in the meeting then. Ok. Ok, bye-bye. [to Pam] Could you please tell Michael that I'm here?
Pam: Sure.

Michael: Hi, Jan. How are you?

Jan: I'm good; how are you?

Michael: Good to see you.

Jan: Nice to see you.

Michael: Ok. [tries to kiss Jan's hand] Ok, why don't we just step into my office? We're gonna go in here.

Jan: Can we please go in your office?

Michael: Yep, right after you. Apres-vous. [mouths to Pam] No calls.

Kevin: Oooo.
Michael: Alright [takes Jan's coat].
Jan: Thank you.

Michael: It's nice to see you.

Jan: Nice to see you too, Michael.

Michael: Really?

Jan: Not like that.

Michael: Oh, well.

Jan: You know Michael, I think I need to make something clear right off the top. I'm not going to discuss anything with you other than Dunder-Mifflin business.

Michael: Alright.

Jan: Period.

Michael: Yep.

Jan: Do we understand each other.

Michael: Absolutely.
Michael: I'm a little confused. 'Cause first it's all like kissy-kissy. And then it's like all regret. Because "Oh, I regret that." But, "Wait, I'm still gonna call you." But, but, "We're just gonna talk business. And I may come down and fire you if you don't do your job." But what were talking about when we first kissed? Business.
Jan: So are you still in the middle of the performance reviews then?
Michael: No, no, no, I finished all of that. I'm very fast. I'm not too fast. Not like wham-bam-thank ya ma'am. But I do say thank ya ma'am. But, I'm, I'm not like wham-bam. Not that there's anything wrong with wham-bam. If it's consensual. [cold Jan stare] We're talking about office stuff. Can I ask you a question?

Jan: No.

Michael: This is a business question. It's nothing personal, I promise.

Jan: Fine.

Michael: Are you wearing a new perfume today?

Jan: How is that a business question?

Michael: Well, you're wearing it at the office. And [smells Jan] it, I'm sorry, but no offense, but it's really sexy.

Jan: Please don't smell me, Michael.
Pam: Hey, Jim.
Jim: Hey, how's it goin'?

Pam: Oh my God, did you see "The Apprentice" last night?

Jim: Course, it's on every Thursday night, so how could I miss it?

Pam: Can you believe who Trump fired?

Jim: No, that was unbelievable.

Dwight: Who? Who was it? Who did he fire?

Pam: You didn't see it?

Dwight: No, I went out and got drunk with my laser tag team last night. Crap! I never go out on a Thursday night; what the hell was I thinking?
Michael: I don't understand- [phone rings] Hold on. Sorry. [answers] Yes, Pam.
Pam: Michael, it's time for the suggestion box meeting.

Michael: I'm kind of in the middle of something. I wish you wouldn't interrupt.

Pam: You told me to buzz you about the suggestion box meeting when Jan was here.

Michael: I did not, not, not use those words.

Jan: Uh, I'd like to sit in on that meeting [to Pam] is it happening right now?

Michael: No, it's in like ten minutes.

Pam: Everyone's waiting in the conference room.

Jan: Great. Very good.
Michael: Why are we here? Because I value your opinions. Now, I know a lot of don't think that I read your suggestions, but I do. I just sift through them every week and I really look and scrutinize to see what you guys are writing. Um, so, let's, uh, just read some of these suckers. Alright. Number one[/b]: "What should we do to prepare for Y2K?"
Dwight: What should we do to prepare for Y2K?

Kelly: I thought you read these every week.

Michael: Well, obviously this one got stuck in the box. [to Jan] That happens occasionally.

Dwight: It happens occasionally.

Michael: And, um, one down. Next suggestion[/b]: "we need better outreach for employees fighting depression". Ok, alright, enough with the jokes. Nobody in here is suffering from depression.

Jan: That sounds serious, Michael.

Michael: Oh, ok, well, yeah, who wrote it?

Dwight: Tom?

Michael: Tom. Then it is a joke because there is nobody in here named Tom.

Phyllis: Tom? He worked in accounting up until about a year ago. [blank stares] Tom? [acts like she's sh**ting herself in the head] Pow.

Michael: Oh, that guy? That guy was weird. Alright, next suggestion.

Dwight: Next suggestion.

Michael: Arrr, dooby dooby do. "You need to do something about your B.O."

Dwight: You need to do something about your B.O.

Michael: Ok, I don't know who this suggestion is meant for, but it's more of a personal suggestion and it's not an office suggestion. Far be it for me to use this as a platform to embarrass anybody.

Toby: Aren't the suggestions meant for you?

Michael: Well, Toby, if by me you are inferring that I have B.O., then I would say that is a very poor choice of words.

Creed: Uh, Michael, he wasn't inferring, he was implying. You were inferring.

Michael: Was I, Creed?! Ok, well, you know what? I am implying is that when we're on an elevator together, I should maybe take the stairs, because talk about t*nk. Not that I would ever say something like that in public, and I never have, and I never will. I just think it's something that we should all be aware of. Ok? Now that we've learned this, let's continue. See, this is good, we're learning and we're figuring some stuff out. "You need to do something about your coffee breath"-

Dwight: You need-

Michael: Ok.

Dwight: To do something about-

Michael: Shut up, shut up, shut up, Dwight, OK. I don't think you people are grasping the concept of the suggestion box.

Angela: Sometimes you talk to us real close.

Michael: Yeah, is that hard for you? Alright, well-

Angela: Well, when you have coffee breath-

Michael: I'll work on that-

Angela: It's hard.

Michael: Let's keep going. Keep it going. Yep. What do we have here? We have somebody's piece of gum. Somebody put a piece of gum in there. This is not a, a garbage can, this is the future of our company. This is not a place for gum. I don't wanna have to read these tomorrow.

Dwight: Yeah, who wants to come in on a Saturday?

Michael: Yeah, what? Uh, alright, next suggestion.

Dwight: Next suggestion.

Michael: "Don't sl-", ok, that's blank [Dwight picks up note] Don't, just put it-

Dwight: "Don't sleep with your boss"? Do you think this is referring to you boning Jan?
Jan: I can't, I can't-
Michael: I don't understand why you're so upset.

Jan: Please sit down.

Michael: Let me ask you-

Jan: You're gonna sit here and I'm gonna go sit over there.

Michael: Ok, let me ask you this.

Jan: Please, sit yourself down.

Michael: Let me ask you something.

Jan: What, Michael.

Michael: Where did you get your outfit?
Dwight: [loud metal music playing in a stairwell; Dwight pacing] You are giving me this raise! I deserve this raise! [plays air guitar] Yes! [kicks] Yes! Yes! Hiya! The least you can do is keep my salary consistent with inflation! Keeya eyah! Yes! Why are you gonna give me this raise? Why? Because... I'm awesome! I am awesome!
Michael: I just don't understand why you have to pretend like nothing happened.
Jan: Because nothing did, Michael. It, I'm not going to say anything more about it, and I would advise that you do the same

Michael: Look-

Dwight: Michael?

Michael: Oh my God...

Dwight: I'm sorry, am I interrupting? Oh God; were you guys making out?

Jan: No, Dwight; come in.

Dwight: Great.

Michael: What do you want Dwight?

Dwight: I am ready for my performance review.

Michael: Ok, great. Your performance has been adequate. You may leave; goodbye.

Jan: Is this how you've been conducting all the reviews, Michael?

Michael: You wanna talk now, good; OK, Dwight, leave.

Dwight: Uh, wait, I would like to discuss my raise?

Michael: Why on earth would we give you a raise?

Dwight: That is an excellent question. Thank you for asking. Let me bring up one word[/b]: dedication. [points to graphs] I have never been late. Also, I have never missed a day due to illness. [Michael sighs] Even when I had walking pneumonia. I even come in on holidays.

Michael: You do? How do you get in?

Dwight: I have a copy of your key.

Jan: That's a serious offense!

Michael: That is a serious offense. Very serious. As is toying with a man's heart.

Jan: Oh! Michael, for God sakes!

Dwight: I'd also further like to talk about my merits in the workplace.

Michael: Ok, third wheel, why don't you do that?

Dwight: For instance, the time I brought in deer jerky for the whole office.

Michael: That was deer!? Gross, oh!

Dwight: You liked it!

Michael: Oh, did not!

Dwight: Jan, have you ever had deer?

Jan: No.

Dwight: It's a delicacy. And you know what? It's an aphrodisiac. So when we're done here, you guys could go over to The Antler Lodge, sample some deer and talk about my raise.

Michael: What do you say, Jan?

Jan: Ok! Here's what I'm gonna do[/b]: I'm gonna step outside, collect my thoughts, and I will return in about ten minutes.

Michael: Ok. You just, uh, clear your head.

Jan: [Dwight opens door] Thank you, Dwight.
Jan: Look, I know it's your job, I know you have to ask, but I promise you, I'm not gonna discuss it with him, I'm certainly not gonna discuss it with you. [digs a cigarette out of her purse] Do you have a light?
Dwight: And in conclusion, I think that Lex Luthor said it best when he said "Dad, you have no idea what I'm capable of".
Michael: That's from Superman?

Dwight: Smallville. And that is why, I feel, that I deserve this raise.
Pam: Do you think Michael and Jan actually...
Jim: I don't really wanna picture it. But thank you, Pam.

Pam: How do you come back from that?

Jim: Um, you don't, I don't think, come all the way back, you know. Especially working together.

Pam: No, I mean doing that with Michael. How do you come back from that?

Jim: Oh-

Pam: As a human being.

Jim: Yeah, no, I don't think you can.
Jan: I'm heading back to New York; Alan and I will conduct your performance review over the phone tomorrow.
Michael: Wait.

Jan: 'K?

Michael: Wait, wait, wait, come, I just, I just wanna know why?

Jan: Michael, now is not the time or the place.

Michael: Ok, so you're saying that there is a different time or place?

Jan: No, I am saying we are never having this conversation.

Michael: Well, ok, well never as in 'never ever ever', or never as in there's still a chance?

Jan: Never, for me, always means 'never ever ever.'

Michael: I just want to know, from the horse's mouth, what is the dealio?

Jan: Michael, it has nothing-

Michael: Am I too short?

Jan: With your looks, ok? It's your personality. I mean, you're obnoxious, and rude, and, and, and stupid, and you do have coffee breath, by the way, and, and I don't agree about the b.o., but you are very, very inconsiderate.

Michael: Really?

Jan: Really. You're, you're, you're a great guy, ok?

Michael: I appreciate that, thank you.

Jan: And you were very sweet, and you stayed up with me and talked with me, cried with me, and I appreciate that-

Michael: No, I wasn't, I didn't cry-

Jan: At this time in my life. I just am not in the place right now where I'm looking for a relationship, so we can still work together, we can still be friends but... ok?

Michael: So my looks have nothing to do with it?

Jan: Ohhh, God.
Michael: Jan is not in a place where she feels she can have a relationship right now. And it doesn't matter how great a guy I am. And that is all I needed; I'm good. I can go home now.
Michael: Hey, it's 12:20; where the hell's Dwight?
Jim: Ummm... no idea.

Michael: Never missed a day, my ass.

Pam: [Jim bows to Pam; she bows back] Thank you.
Dwight: [running through parking lot] I'm here! I'm here! I'm here! It's ok!
Michael: It is Friday morning and it is another beautiful day in Scranton, Pennsylvania. [sees man in a turban outside] Oh my God. Ohhh. [dials phone number] Pick up, pick up, pick up, pick up. Oh, we have a serious problem here. [goes out onto office floor] Alright everybody, lock the doors, turn off the lights. Pretend you're not here.
Jim: Are we in danger?

Michael: There's no time to think about if this is real. Just, shh, everybody. [knock at the front door]

Kevin: Michael, should I call the... [Michael waves his hands] What?
Michael: The IT tech guy and me did not get off to a great start.
Michael: Yeah, I tried to install it myself, but, uh, you guys have these things so password-protected...
Sadiq (IT guy): That just means you have to enter your password.

Michael: Oh...

Sadiq (IT guy): What's your password, Michael?

Michael: Oh, umm... [looks at Post-It on computer]

Sadiq (IT guy): Oh, it's 1-2-3.

Michael: Yes.
Dwight: Mi-
Michael: AH! Guh-oood.

Dwight: Sorry.

Michael: Please don't do that.

Dwight: Ok, I'm sorry. What is going on in there? Why is he here? What are you doing?

Michael: I can't tell you.

Dwight: You have to tell me.

Michael: I don't have to tell you anything.

Dwight: Look, Michael, I know you don't want to have to think about this, but if something were to happen to you, God forbid, then I would need to know in order to take over.

Michael: Dwight, nothing is going to happen to me, ok? I'm in the best shape of my life. Look at this. [flexes his arms] Brrr! That's strong!

Dwight: Yeah, but that doesn't matter, you could get a brain aneurysm-

Michael: I'm not going get a brain-

Dwight: Or get hit by a car-

Michael: Stop it.

Dwight: Or a bus or a train. Get poisoned, fall in a well, step on a mine, choke.

Michael: Uh, oh, ok; if I step on a mine in Scranton, Pennsylvania, and die, you can have my job, ok? Why don't you just go... away?
Michael: There are certain things a boss does not share with his employees. His salary, that would depress them. His bed, it--- And I am not going to tell them that I'll be reading their e-mails.
Michael: So how do you search?
Sadiq (IT guy): By keyword phrase.

Michael: Try "profits". No! Try "Michael Scott". "Michael" "boss" and "funny". [Sadiq (IT guy) types; result pops up] Oh my God, wow! [chuckles] E-mail from Stanley. Stanley, Terribly nice guy. [reads e-mail] "Sorry I didn't write back sooner; I can't go to the game tonight because my boss Michael is an ass and making me stay late." Well, Stanley's an ass. Not one of our harder workers.
Oscar: Hey, what's the deal, Michael? Why are you spying on our computers?
Michael: Oh, no, everybody; Oscar's gone crazy! What other ghost stories do you have for us? That I'm a robot? [robot voice] I will destroy everything in my path-

Oscar: Actually, it's just-

Michael: Beep! Bop!

Oscar: Ok...

Michael: Bommmm. Bop! Onk onk. [Tin Man voice] Oil can. Oil can.

Oscar: Tin Man. Actually we just a got a memo from IT saying you're doing e-mail surveillance.

Michael: Oh, what? No. That defeats the whole purpose.

Dwight: So it's true? You have access to our e-mails?

Michael: You know what the problem is?

Stanley: I think I do.

Michael: The problem is that when people hear the term "big brother", they immediately think it's scary or bad, but I don't. I think, wow, I love my big brother.
Kevin: I gotta erase a lotta stuff. A lot of stuff.
Dwight: Oh hey, just so ya know, if you have a lot of sensitive e-mails, they need to be deleted immediately.
Angela: I know.

Dwight: Good. [Pam overhears]

Pam: [whispers to Jim] Hey. Something just happened. Dwight just told Angela that she has to deleted all of her sensitive e-mails immediately.

Jim: What?

Pam: I know!

Jim: Hmm...

Pam: Do you think that they're like-

Jim: No.

Pam: No, right, no, no.

Jim: [humming]

Pam: Uhhh, ew, ew, ew... .Maybe?
Pam: It's like squishing a spider under a book. It's gonna be really gross but I have to look and make sure that it's really dead. Sooo... [to camera guys] If you guys see anything... ?
Pam: Hey, Dwight, um, my friend is kinda into these two girls that he works with.
Dwight: Nice.

Pam: One is tall and brunette, and the other one is short, and blonde, and perky, and kinda judgmental. Who do you think he should choose?

Dwight: Does he have access to their medical records?

Pam: Ummmm...
Dwight: I think one of the greatest things about modern America is the computerization of medical records. As a volunteer sheriff, I can look up anyone's psychiatric records or surgical histories. Yeast infections. There are a huge number of yeast infections in this county. Probably because we are down river from that old bread factory.
Michael: Meredith has an E-vite from Jim. Barbeque at Jim's tonight. Tonight? Wonder where my e-vitation is. Click on guest list. Angela, Stanley, Oscar, Meredith, Phyllis, Kevin, Creed. Must be... .[scrolls down list] No.
Pam: Hey, Angela-
Angela: Hi.

Pam: How's it going?

Angela: It's ok.

Pam: Listen, are you bringing anyone to Jim's party tonight?

Angela: No. Are we supposed to?

Pam: No. I mean, I don't know, I don't think so.

Angela: Hmm... [Pam reaches towards vending machine] Excuse me.

Pam: Oh.
Michael: There's always a distance between a boss and the employees. It is just nature's rule. It's intimidation mostly, it's the awareness that they are not me. I do think that I am very approachable, as one of the guys. But maybe I need to be even approachabler.
Kevin: That's pretty young.
Pam: Yeah.

Kevin: [to Michael] Are you gonna eat with us?

Michael: Of course. Hangin' with my crew, crew that I am one of. Hangin' with my Cup of Noodles. This is a meal in a cup.

Jim: Uh hum.

Michael: Hot, tasty. Reminds me of college. Lived on this stuff. Brain food. Mmmm... You know what I really, really miss about college? The parties. Everybody'd go. The athletes, the, the nerds, professors.

Pam: The professors would go to the parties?

Michael: Yeah! They were the most fun. We always invited them.
Jim: It's true. I'm having a party. I've got three cases of imported beer, a karaoke machine, and I didn't invite Michael. So three ingredients for a great party. And it's nothing personal, I just think that if he were there, people wouldn't be able to relax, and you know, have fun, and my roommate wants to meet everybody. Because I'm pretty sure he thinks that I'm making Dwight up. [sighs] He is very real.
Pam: [to cameraman] What? [looks at Dwight eating a Baby Ruth] Ohhhh... Yes! Thank you!
Dwight: Question[/b]: on the Internet there are several different options to get to your house for the party tonight-
Jim: Oh, uh, no. Could-

Dwight: I was wondering-

Jim: Could, keep that down.

Dwight: Why?

Jim: Because not everybody knows about the party.

Dwight: Like who? Who doesn't know?

Jim: Umm, Michael.

Dwight: Why just Michael?

Jim: Because it's a surprise.

Dwight: Is it?

Jim: Uh hmm.

Dwight: Oh, that's perfect!

Jim: So, don't tell.

Dwight: I won't.

Jim: Ok.
Jim: So, Dwight thinks that tonight is a surprise party for Michael.
Pam: Really? That's great.

Jim: I know.

Pam: Maybe we can get him to hide and wait somewhere.

Jim: [laughs] Oh man. Oh, you know what, speaking of which, I was just trying to get a handle on, you know, numbers for food and stuff. So do you think that Roy's gonna come, or...

Pam: Oh, no, he can't make it.

Jim: Oh, ok, cool.

Michael: Hey there.

Jim: Hey.

Michael: Almost quittin' time.

Jim: Yup, it's, uh, four o'clock.

Michael: One more hour. Take care of anything you forgot to do. Hey, you know, I don't know if you have any plans tonight, but if ya don't, we could hang out.

Jim: Oh, um... .I can't.

Michael: You have plans.

Jim: Uh hmm, definitely.

Michael: I do, too. I do, too.

Jim: You do?

Michael: I do, yeah. Big plans.

Jim: Because you said "do you wanna hang out"-

Michael: Tonight, I can't do it tonight, no. Improv class, I have improv class, hanging out with my improv buds-

Jim: Really?

Michael: Yeah.

Jim: Aw, that sounds like a lot of fun.

Michael: It's the best. It's the best. I would not miss it for the world. But if something else came up, I would definitely not go.

Jim: Improv sounds great.

Michael: It is. Ok.

Jim: Alright.

Michael: [someone coughs] What?

Jim: I think Stanley just coughed.
Michael: Hey, Pam. Do you need me to walk to your vehicular transport?
Pam: No thanks.

Michael: Alright. Oscar, got big plans tonight with-

Oscar: I'm on a call.

Michael: Kevin, big man, big man, what are you doing tonight? Where are you off to?

Kevin: My brother is in town and we are going to see the Alaska Film Festival at the Scien-

Michael: Ok, alright.

Kevin: Oh.

Michael: Hey, Angela, rushy, rushy. Where you rushin' off to?

Angela: I'm just leaving for the day.

Michael: Yeah, well duh. Where ya headed?

Angela: Charity. Bake drive.

Michael: Liar!

Angela: No!

Michael: You are a liar.

Angela: No, I'm not.

Michael: Dwight, oh ho, Dwight, Dwight, my loyal compadre. You and I are hangin' tonight. The two of us. We are celebrating our freedom and our manhood. You know what? Why don't we watch that show that you've been wanting to watch, that stupid Battleship Galaxy.

Dwight: Battlestar Galatica.

Michael: That's, whatever stupid show you want-

Dwight: I can't-

Michael: To watch tonight

Dwight: Tonight.

Michael: We're watching it.

Dwight: Unfortunately, I've got plans. I have to go to practice. Soccer practice.

Michael: I didn't know you played soccer, Dwight.

Dwight: Clarinet.

Michael: You, too, Dwight?

Dwight: Have fun tonight! Whatever it is that you're doing, and I'll see you Monday. [to the camera] He has no idea!
Jim: Quick announcement everybody, if I could have everybody's attention. We do have wine in the kitchen, and, uh, there is beer available on the porch and despite what you might think, it's not all for Meredith and Kelly, so please enjoy.
Dwight: Jim! You really think this is a good idea, huh? A hide-a-key rock?

Mark: Hey, you must be Dwight!

Dwight: You don't work with us.

Jim: That's because Mark's my roommate.

Mark: Hey, I love the Birkenstocks.

Dwight: Thanks. Yeah, I always keep an extra set in the car, for special occasions. Jim, come here.

Jim: Uh hmm.

Dwight: When is the guest of honor coming?

Jim: Oh, uh, later-ish.

Dwight: He's gonna love it!

Jim: Great. [to a group of guests] Just wanna let you guys know that we will be taking the tour like I promised-

Pam: Hey.

Jim: Hey! Just in time! You wanna go on the group tour? We were just about to leave.

Pam: Definitely.

Jim: Well, the group tour is now leaving, then. Ladies and gentlemen, just a few things that we are gonna be pointing out to you today, you will be able to see both bedrooms and, uh, if we're lucky, maybe get a chance to peek into the bathroom, who knows. I have to remind you that flash photography is prohibited and as much as you can, please refrain from touching things. I know you might want to.

Ryan: Hey, is Katy coming?

Jim: Uh, actually, I haven't talked to her in a while.

Ryan: Huh, is it ok if I call her?

Jim: We can talk about that later.
Improv Teacher: Ok, let's get right into it. I need two people for the first scene.
Michael: [In Horshack voice from 'Welcome Back, Kotter'] Ooo Ooo Ooo Ooo, Mr. Cart-air, Mr. Cart-air.

Improv Teacher: Ok, Michael. And... anybody? How about Mary-Beth? Come on. Ok, so you start us off Mary-Beth.

Mary-Beth: Great. [skips] La la la la la laaa...

Michael: [fake kicks in door] Boom! Detective Michael Scarn. I'm with the FBI.
Michael: Think about this; what is the most exciting thing that can happen, on TV, or in movies, or in real life? Somebody has a g*n. [gasps] That's why I always start with a g*n, because you can't top it, you just can't.
Girl acting Pregnant: I'm supposed to meet my doctor here? Have ya seen him? He's a very angry midget.
Michael: Boom! Freeze! Michael Scoon, FBI. You know what you did. Boom! Boom! Boom! [sh**t at Pregnant Girl and another actor] Yeah, you thought that you could get away with your little ruse, didn't you? Didn't ya!? Well, you didn't, because I know where ya hid the diamonds. I been on to you and your little friends for weeks. [another actor steps in] Boom! Boom! Boom!

Actor: I'm not even in the scene!

Actress: Again!?

Michael: Boom! Boom!

Improv Teacher: Stop, stop, ok, stop.

Michael: Boom! Boom!

Improv Teacher: You sh*t me, great. Now stop.

Michael: Why

Improv Teacher: You can't just sh**t everyone in the scene.

Michael: Well, if you hadn't stopped the scene, you would have seen where it was going.

Improv Teacher: Ok, what about the scene they set up?

Michael: Boring.

Improv Teacher: No, it wasn't. No more g*n.

Michael: I could of-

Improv Teacher: No. No. Michael, I want you to give me all the g*n you have.

Michael: Ok.

Improv Teacher: Just, I want you to get rid of all your g*n and give them to me. Great.

Michael: Yehhhehh.

Improv Teacher: Yeeehh, ok.
Pam: Jim's bedroom.
Jim: See, I knew we lost somebody on the tour. It's-

Pam: Cool... This is your desk.

Jim: This is my desk.

Pam: Your home office?

Jim: My home office, this is it.

Pam: Down. You have to sit down so I can get the full effect.

Jim: Ok, sure. Will do.

Pam: Ok, wait, so that would make me [walks to side of room] like right here.

Jim: Yeah, that... Yep, that feels about right.

Pam: And Dwight would be like-

Jim: You know what? Let's just leave that image out of it, because this is a happy place. Happy thoughts, Pam. Happy thoughts.

Pam: Umm, [gasps] yearbook!

Jim: Yeah, you don't have to, ummm. Alright, yes, that's not gonna be awkward at all.

Pam: [finds Jim's picture] Ooooohhh no!

Jim: Oh yeah.

Pam: You were so dorky!

Jim: Thank you.
Improv Teacher: Freeze!
Michael: I'm in.

Improv Teacher: You wanna go over the rules one more time?

Michael: No, no, no. I'm looking in my wallet for money so you can tell me my fortune.

Bill: I promise it's worthit . Ooo, I can see you walking out of here and you're thrilled with your reading.

Michael: What are you... [whispers to Bill]

Improv Teacher: Michael, what did you tell him?

Michael: Nothing.

Improv Teacher: Then why are his hands up? Bill?

Bill: He told me he couldn't show it to me, but he has a g*n.

Improv Teacher: Ok, let's call it a day. Nice job, Bill.

Michael: Good, it's good. Good work, everybody.
Jim: Angela, burger? Dog? Havin' fun?
Angela: I got sap on me.

Jim: Chicken, hot dog, burger.

Angela: I'm a vegetarian.

Jim: There is soda inside.

Angela: Guh.
Stanley: I didn't think the premium laser color copy batch would sell as well as it did.
Oscar: Yeah, it surprised us all. I'll tell you why. Because when they---

Kelly: I'm sorry guys; can we please not talk about paper? There's gotta be something else we can talk about.
Angela: I think it's alright. Jesus drank wine. [Pam overhears]
Pam: Hey Phyllis, come here for a second.

Phyllis: Sure.

Pam: Have you heard anything about any secret office romances?

Phyllis: You tell me. Well, you do mean you and Jim right? Oh God. I am so sorry, I thought, you guys hang out all the time and you're talking all the time. I'm sorry!

Pam: That's ok. It's ok.
Kevin: [smacks Ryan's hand] Not so fast... Fire Guy.
Mary-Beth: How do I get to Bernie's Tavern from here?
Bill: Oh, don't worry. We're all gonna carpool.

Michael: So Bernie's, huh? We're all going to Bernie's? [to camera] Go to Bernie's?

Bill: Oh sorry, we're not going as a group, it's just a private friend who just happens to know all of us from different ways is throwing a private birthday thing.

Michael: Right, right, right. Well guys, I'd love to go to Bernie's with you, but, you know, I have an office party. A big office party I need to go to, so... Can't get out of it.

Bill: Ok.

Michael: Ok, see you later. Nice job, Bill. Not.
Phyllis: [singing] Here I go again on my own. Going down the only road I've ever known...
Jim: Oh by the way how's your side project going?

Pam: Oh, yeah I gave that up.

Jim: Really?

Pam: Yeah, it turns out I was, um, just grasping at straws.
Pam: Just because two people are hanging out, it doesn't mean that they're together, you know? Like people can just be friends. And I think that it was really unfair to think that there was anything else going on.
Kevin: [singing] Just turn around now, cause you're not welcome anymore.
Dwight: Surprise! [laughs] Everybody!

Phyllis: Dwight...

Michael: Wow! Who opened the morgue for this thing? I'm just driving by, thought I'd drop in. [to Jim] There's some wine. I would love a glass, if you're gonna open it. Here ya go Temp, take my jacket! [sees Sadiq (IT guy)] Oh, come on! That guy? He is a good guy, not a t*rror1st. Karaoke, I love it! I am a karaoke fiend. I call dibs. I got next, I got next up. Come on, let's get this party started, ha! Ok? Where's that wine?
Michael: Ok, yeah, this is a duet, so, need somebody else, Pam? You wanna come up and sing this one? Need somebody else. Takers, please. [singing] Baby when I met you there was peace unknown. Kelly? Tried to get you with a fine tooth comb. I was soft inside, there was something goin' on. This part goes to the, uh, girl. You do something to me that I can't explain. Hold me closer and I feel no pain. In every b*at of my heart, we got something goin' on. Tender love is blind---
Michael and Jim: [singing] It requires a dedication, all this love we feel needs no conversation. Divided, together, uh huh. Making love with each other, uh huh.

Michael: We're making love!

Michael and Jim: [singing] Islands in the stream, that is what we are, no one in between, how can we be wrong? Sail away with me...
Michael: Talk! Just talk!
Mary-Beth: I am-

Michael: Shut up!
Michael: Funny story[/b]: the way that I got into improv was, I got into improv. The story about me getting into improv was that I was walking down the street, and a race car pulls up, and the guy says "Hey you're funny, you're the funniest guy I've ever seen, or my name is not Dale Earnhardt." [giggles] And that was an improv. Um, the real way is that I found a flyer.
Dwight: Go. Get the door.
Michael: Here we are.

Dwight: Go. Push!

Michael: Oh god.

Dwight: Push!

Michael: No, no, turn it around.

Dwight: Really shove it.

Michael: You'll break it.

Dwight: Shove it through! Break it!

Michael: You shove it. Shove it back! Here we go. Don't break the branches, Dwight.

Michael: All right.

Dwight: I got a splinter.

Michael: Well, suck it up. We all have problems. Hey, everybody, look what we have! [laughs] Nice, huh?

Dwight: I've got it leveraged. Push. Straight up.

Michael: On three. Ready? Big, one, two, three.

Dwight: One, two, three. [they push the tree up and it breaks through a ceiling tile.]

Michael: Merry Christmas!
Michael: Did it work?
Kevin: [holds up the piece of tree he just cut off with a paper cutter] Well, sort of. Why did you get it so big?

Michael: A, that's what she said, and B, I wanted it to be impressive. The biggest day of the year deserves the biggest tree of the year.

Kevin: But what are we going to do with this hacked off part?

Michael: Well, that is a perfectly good mini-tree, Kevin. And we are going to sell that to charity. That's what Christmas is all about.
Jim: So this year, for the first time ever, I got Pam in Secret Santa. And I got her this teapot, which I know she really wants, so she can make tea at her desk. But I'm also going to stuff it with some inside jokes. Like, this is my high school yearbook photo. She saw it at the party, and it really makes her laugh. Not sure why. What else .. ooh. This is a hot sauce packet. She put this on a hot dog a couple years ago because she thought it was ketchup. And it was really funny, so I kept the other two. [holds up a miniature pencil] This would take a little too long to explain, so I won't. And this is the card. Because Christmas is the time to tell people how you feel.
Angela: Is there anything we're missing? Phyllis, you got the lights?
Phyllis: Yes, I got those cute little ones. [Angela looks at her disapprovingly] Do you think I should have gotten the big ones?

Angela: We'll see.
Ryan: Angela drafted me into the party planning committee. Her memo said that we need to prepare for every possible disaster. Which to me seems excessive.
Michael: [comes into the conference room in a Santa hat and beard] Merry Christmas! Ho, ho, ho, [points to Ryan] pimp. I'm kidding. What do we got, what do we got? How many plates are we getting?
Angela: Fifty.

Michael: Double it. Double everything. Double ice cream. Double napkins. Double it. On me.
Michael: It was a tough year. I had to fire somebody this year. This party has to really rock. Check it out. Christmas bonus. 3,000 G's. I got this for helping save the company money. So I guess some good came out of firing Devon after all. Maybe I should call him and tell him that.
Michael: I want people to cut loose. I want people making out in closets. I want people hanging from the ceilings, lamp shades on the heads. I want it to be a Playboy Mansion party. And also, I want you to spread the word that I will have my digital camera. And I'll be taking pictures all along the way. And the best and craziest thing that happens will be on the cover of the newsletter. Incentive.
Pam: You do realize that we can't serve liquor at the party.

Michael: Yeah, I know. Damn it. Stupid corporate wet blankets. Like booze ever k*ll anybody.
Kevin & Oscar: One, two, three. [they lift and start to move a desk]
Dwight: You guys should use a hand truck.

Kevin: Do we have one?

Dwight: No.
Angela: [having trouble with a plastic tablecloth as Pam stands idly by] Will you help me?
Michael: No! No way! It... no.
Darryl: Come on, Mike, let me borrow the hat for just a couple of hours.

Michael: You wanna be Santa?

Darryl: Yeah.

Michael: Have you ever seen Santa?

Darryl: Yeah, I've seen Santa.

Michael: Okay.

Darryl: Who cares?

Michael: Well, I'm sorry. It just doesn't work.

Dwight: Michael, I would like to be the elf.

Michael: That makes sense because he has elfish features.
Dwight: [now wearing an elf hat and ears] Okay, everybody listen up! It is time to get your presents, wrap them, and place them under the tree like so. If you do not get your present wrapped and under the tree within the next five minutes you will be disqualified from Secret Santa. All right? No exceptions except Michael.
Toby: I got Angela. She is into these posters of babies dressed as adults. I got her one of those. I felt kind of weird buying that.
Oscar: I got Creed. And to tell you the truth, I don't know anything about Creed. I know his name's Creed. I know he works right over there. I think he's Irish and I .. I got him this shamrock keychain.
Kevin: I got myself for Secret Santa. I was supposed to tell somebody, but I didn't [smiles happily].
Michael: You get something good this year?
Jim: I think I did a pretty good job.

Michael: Yeah? Who did you have?

Jim: Well, I can't tell you cause it's a secret.

Michael: I think I got something pretty nice for my guy.

Jim: Yeah?

Michael: I spent a lot of dough. Lot of dough. Wow.

Jim: Well, there's a $20 limit, right? So .. ?

Michael: Yeah. I wanted this party to be really special so I sorta went above and beyond.

Jim: That's great. Well don't tell me who it is, cause I can ..

Michael: It was Ryan. Yeah. I have Ryan.
Dwight: Gather round. Secret Santa, let's go. Let's go. Come on. Stanley, no, I'm going to handle the cord. Okay, safety reasons.
Stanley: I know how to plug something in.

Dwight: I want to do it.

Michael: All right, let's count it down, like Rockefeller Center. Ready?

All: Three, two, one. [very dim lights come on the tree]

Michael: Not great.

Phyllis: I'm sorry, everybody.

Pam: I think the tree looks nice.

Dwight: Hey, I could get some flares from my car.

Michael: No, no. Shake it off, everybody. Just, let's do Secret Santa.
Michael: Presents are the best way to show someone how much you care. It's like this tangible thing that you can point to and say, "Hey, man, I love you this many dollars worth."
Dwight: First present, Oscar.
Oscar: [rips off the wrapping] Shower radio. Neat.

Kelly: Oh, good, that was from me.

Oscar: Thanks, Kelly. You know I was gonna get one of ..

Dwight: Okay. Okay. That's enough. Let's keep it moving on. Jim.

Jim: Oh, cool. [opens his plastic bag]

Creed: That's from me.

Jim: Great. Where did you get it?

Creed: I don't know. It was so long ago.
Jim: He obviously forgot to get me something, and then he went in his closet and dug out this little number [holds up way-too-short sleeves] and then threw it in a bag.
Creed: Yep. That's exactly what happened.
Dwight: Pam.
Pam: [opens up her present] Oh, my god! Thank you very much, Santa, whoever you are. It's awesome.

Jim: There's a little more to it.

Dwight: All right, next. Ryan. [tosses present]

Michael: No, don't!

Ryan: [unwraps present] Whoa, a video iPod.

Michael: Whoa. Wow. Jeez. Somebody really got carried away with the spirit of Christmas. That was me, I got a little carried away.

Ryan: Wasn't there a $20 limit on the gift? This is 400 bucks.

Michael: You don't know that.

Ryan: Yeah, you left the price tag on.

Michael: I did?

Ryan: Yeah.

Michael: What? Oh, sh**t. Wow. Okay, well, who cares? It doesn't matter what I spent. What matters is that Christmas is fun, right?

Dwight: Michael.

Michael: Oh hey, for me. What is in here? [opens a handmade oven mitt] Oh, come on.

Phyllis: I knitted it for you.

Michael: An oven mitt? Okay. [walks out]
Michael: So Phyllis is basically saying, "Hey Michael, I know you did a lot to help the office this year but I only care about you a homemade oven mitt's worth." I gave Ryan an iPod.
Kevin: Should we just keep opening up the presents?
Dwight: We don't do anything until Michael gives us further instructions.

Michael: I got it! We are going to turn Secret Santa into Yankee Swap.

Jim: What is Yankee Swap?

Michael: One person chooses a gift. The next person can either choose a gift or steal that person's gift. If your gift gets stolen, then you can steal somebody else's gift or choose a new gift.

Jim: I thought that was called Nasty Christmas.

Pam: Yeah, we call it White Elephant.

Michael: Well, I call it fun!

Oscar: Why are we doing this?

Michael: Because it's better. Because it's more special.

Angela: It sounds mean.

Michael: Shut it. No, it's not. Okay, just give it a sh*t.
Angela: Michael should have asked the party planning committee first. He's not supposed to just spring things on us out of nowhere. [starts to cry]
Michael: Okay, Meredith is up first. Here's the deal. You can either pick a new gift or you can steal somebody else's gift that they've already gotten, like the oven mitt.
Meredith: I'll take the teapot.

Jim: Oh, shouldn't we ... I bought that specifically for Pam.

Michael: Yankee Swap! That's what makes it fun. Pam, you can steal the oven mitt now.

Pam: I'll take the iPod.

Ryan: And I have to give it to her? I don't have a choice?

Dwight: Yes, now you can steal the oven mitt, the old shirt or the shower radio or pick a new gift.

Stanley: [after Ryan opens a new gift - a nameplate saying 'Kelly'] That was meant for Kelly.

Ryan: Yeah, I figured.

Michael: I think this is going great.
Kelly: [unwrapping the poster] Yikes.
Toby: Well, it's for Angela, so ..

Kelly: That's like, the creepiest thing that I've ever seen.

Dwight: Angela, you're up.

Angela: I'll take the poster. Some people like these.

Kelly: I will steal the iPod.
Michael: Everyone wants the iPod. It's a huge hit. It is almost a Christmas miracle.
Michael: Oh, well, Oscar, you little gourmand, you have the next turn.
Oscar: I'll take the ... teapot.

Meredith: Damn it.

Dwight: Okay, moving along. Meredith, let's go.

Meredith: I really want the iPod.

Dwight: It's already been stolen this round. Pick something else.

Michael: [holding oven mitt] I hope nobody takes this baby, cause this is great. Wow, look at that fine craftsmanship. Somebody really put a lot of work into that. It's beautiful.

Meredith: I'll take the oven mitt.

Michael: Sucker! See, I wanted somebody to take it. Boom! Reverse psychology.
Michael: Reverse psychology is an awesome tool. I don't know if you guys know about it, but basically you make someone think the opposite of what you believe and that tricks them into doing something stupid. Works like a charm.
Michael: [opens present] "In addition to these paintball pellets, your gift includes two paintball lessons with Dwight Schrute."
Dwight: You and me, Michael. Yes!

Michael: Who wants to take paintball lessons? How is that better than an iPod?

Dwight: I never said it was better than an iPod.
Dwight: Michael keeps bragging about his iPod, but you know what? Two paintball lessons with someone as experienced as I am is worth easily, like, 2 grand.
Dwight: [sh**ting paintball g*n at target] Take that, Saddam!
Michael: Last gift. Kevin.
Kevin: I want the foot bath.
Kevin: That's the thing I bought myself. I'm really psyched to use it. [pauses] Maybe I should have taken the iPod. Oh, sh**t.
Dwight: Pam, steal something or pick the final gift.
Pam: I want the iPod.

Kelly: Damn it.

Jim: Sure you don't want the teapot?

Pam: Well, I mean, it's an iPod. But ..

Jim: Right.

Pam: Sorry, I ..

Jim: No. No. Definitely. It's ..

Kelly: Okay, well, I guess I will take that book of short stories.

Dwight: Yes! There you go. I want the teapot. Gracias.

Jim: Got to be kidding me.
Dwight: Yankee Swap is like Machiavelli meets ... Christmas.
Michael: [after Phyllis leaves suddenly] What is she so upset about?
Pam: Maybe because you hated her present so much.

Michael: Come on! I think that Yankee Swap was a big hit! I think it's a success and I'm the one who ended up with Dwight's stupid paintball pellets.

Jim: Yeah, but, Michael, the point is that we all bought gifts for specific people.

Stanley: And you should have just bought a $20 gift like everyone else.

Michael: Well, I didn't. I got a big bonus because I fired Devon, and I used the money to buy something awesome. Sue me!

Oscar: You got a bonus check?

Pam: How much?

Michael: It wasn't. It wasn't that much. It was $3,000.

Stanley: All right, I'm done now.
Michael: Unbelievable. I do the nicest thing that anyone has ever done for these people and they freak out. Well, happy birthday, Jesus. Sorry your party's so lame.
Liquor Store Clerk: It comes to $166.41.
Michael: All right, now, you're the expert. Is this enough to get 20 people plastered?

Liquor Store Clerk: Fifteen bottles of vodka? Yeah, that should do it.

Michael: Cool, cool. Box it up.
Jim: I bought this teapot for Pam, and I know she really wants it. So, can I trade you for it?
Dwight: No trades.

Jim: Come on, it's a shamrock keychain. Good luck.

Dwight: "A real man makes his own luck." Billy Zane. Titanic.

Jim: Look, it has sentimental value, Dwight. Can I buy it from you?

Dwight: No. I want it. I'm going to use it.

Jim: You don't even drink tea.

Dwight: True. But I get sinus infections, and sinus infections can be cured by making your tea from green tea leaf stems ..

JIm: Okay ..

Dwight: .. and pouring it directly into your nose, like so. [demonstrates]
Jim: To think that my gift for Pam will be used for that, it's a little too much to handle.
Roy: This is awesome.
Pam: I know. It's totally going to change the way I work out.

Roy: Yeah, I was gonna get you one of these for Christmas, and now I don't have to. I'm gonna save a ton of money.

Pam: So what are you going to get me instead?

Roy: I don't know. Probably like, a sweater or something.
Michael: Uh-oh. Looks like Santa was a little naughty.
Angela: What is that?

Michael: This is Christmas spirit, as in spirits, booze.

Meredith: We can drink?

Toby: We're really not supposed to serve alcohol.

Michael: Zip it, Toby! Just .. I mean, it's a party. Come on. If I can't throw a good party for my employees, then I am a terrible boss. Who wants a drink?

Meredith: Me. Please.

Michael: Go, here we go!
Meredith: The deal is that this is my last hurrah, cause I made a New Year's resolution that I'm not going to drink anymore. During the week.
Phyllis: Hi guys.
Ryan: Hey.

Phyllis: Does everybody know my boyfriend, Bob Vance?

Kevin: Kevin Malone.

Bob Vance: Bob Vance, Vance Refrigeration.

Stanley: Stanley Hudson.

Bob Vance: Bob Vance, Vance Refrigeration.

Ryan: Ryan Howard.

Bob Vance: Bob Vance, Vance Refrigeration.

Ryan: What line of work you in, Bob?
Roy: I think after I lost Culpepper and T.O it was over, man.
Darryl: Oh, yeah, I need McMahon, Deion Branch to have big games or else I'm done.

Roy: It's possible. I can't believe you traded Shaun Alexander, man.

Darryl: I had to. I needed defense.

Roy: Come on! Shaun Alexander? He's the best back in the league.

Darryl: It's defense.

Roy: Oh, no. That is not worth it.

Darryl: It is worth it.

Roy: Never.

Darryl: Are you kidding? You wait.
Michael: Anybody making out in here? [checks hallway] Not yet, give it time. Oh, hey, Ebenezer, boink. [takes picture of Jim] Okay, how's it going in here? [takes picture of Meredith and Kevin]
Ryan: We're running low on cups. Do you want me to just run out and get some?

Angela: There should be some ..

Michael: No, no, no, no. We'll find some, don't leave the party.
Phyllis, Meredith, Michael, Kevin: One, two, three. [do a sh*t]
Michael: Kudos to Ryan, king of the party committee.

Ryan: Oh, no.

Michael: Yeah, yeah, yeah.

Ryan: I really did not do anything.

Michael: Oh, no, no. No false modesty, my friend.
Jim: You know, you don't have to answer calls during a party. Just thought you should know.
Pam: [laughs] No, I was just checking out my present. [holds up teapot]

Jim: But ..

Pam: I traded with Dwight. Just, I figured, you know, you went to a lot of trouble and it means a lot. And also, Roy got me an iPod or was going to get me an iPod, so ..

Jim: Well, either way. This is an amazing gift because it comes with bonus gifts. Look inside.

Pam: [opens teapot] Oh my god! The yearbook picture!
Pam: Yeah, I think I made the right choice.
Pam: Oh, my God! It's incredible. [Jim reaches and steals his card back before she can notice it] Is this the Boggle timer?
Jim: I didn't think you were going to get that one. I really didn't.
Dwight: This is so awesome.
Michael: Not bad. And if it couldn't go to Ryan, you are the guy I'd want it to go to.

Dwight: Thank you.

Michael: You're welcome.

Todd Packer: [grabbing Michael around the neck] Merry Christmas, asswipe!

Michael: No way. Oh, you're kidding me. Packer! Yes! Todd Packer, ladies and gentlemen!

Todd Packer: [rapping] What's up my nerds. Check it out. [points at the mistletoe stuck down his pants]

Michael: Oh, no, no. Oh look at that. Icing on the cake.

Todd Packer: Pacman need a drinky.

Michael: Oh, let's fix you up. Who wants to fix up .. Toby. Toby's gonna fix you up.
Kevin: [listening to music through headphones] Yeahhh.
Michael: Darryl. There you go. [hands him the Santa hat] You earned it.
Darryl: That's okay, Mike.

Michael: No, no, no, no. I really, really want you to have it.

Darryl: All right. Thanks, man.

Michael: Hey, Merry Christmas.
Ryan: [looking at Xeroxed butt pictures] Whose butt is that?
Kevin: Mine.

Ryan: Oh, how did I not guess that?
Michael: [coming out of his office] Lampshade on head! It's happening!
Creed: [as Jim decorates a passed out Todd Packer] Oh, no.
Kelly: Hey.
Dwight: Oh, hello there. [Kelly leans up and kisses him] But what are you doing?

Kelly: I don't know.

Dwight: You shouldn't do things like that. The man is supposed to do that.
Kevin: Thanks for the party, Michael.
Meredith: Yeah.

Bob Vance: Oh, hey. Listen up. We're going to Poor Richard's. Who's in?

Oscar: I'm in.

Dwight: Yes.

Oscar: Michael? Poor Richard's?

Michael: Yeah, that sounds good.
Michael: Christmas is awesome. First of all, you get to spend time with people you love. Secondly, you can get drunk and no one can say anything. Third, you give presents. What's better than giving presents? And fourth, getting presents. So, four things. Not bad for one day. It's really the greatest day of all time.
Michael: Hey, Meredith. Heading over to Poor Richard's?
Meredith: Yep.

Michael: Cool, cool, cool. Do you need a ride? [Meredith drops her top. Michael takes a picture] All right, let's head out. Sounds good. Do you have a coat?

Meredith: Yeah.

Michael: Okay!
Jim: Hey.
Dwight: Hello. Jim?

Jim: What's up, buddy?

Dwight: This is not funny. Why is my stuff in here?

Jim: Wow, that's weird. Oh, dollar for a stapler, that's pretty good.

Dwight: Yeah, well, I'm not paying for my own stuff, okay? I know you did this, because you're friends with the vending machine guy.

Jim: Who, Steve?

Dwight: Yeah, Steve, whatever his name is.

Pam: Sorry. What do I want? What do I want... Oh, it's a pencil cup.

Dwight: No, no, no, no, no. That's my pencil cup.

Pam: Um, I don't think so, I just bought it.

Dwight: Uh, I think so, and you're going to hand it over to me.

Pam: I love these.

Dwight: Okay, fine. Where's my wallet?

Jim: Oh, there it is. J1.

Dwight: But I don't have any...

Jim: Here, you know what? You can have some nickels.

Dwight: [putting quarters in] Five, ten, fifteen, twenty, twenty-five...
Michael: Hello, everyone.
Dwight: Good morning, Michael.

Phyllis: Where are we going this afternoon?

Michael: Ah! Ha ha ha!
Pam: Last week, Michael sent out this mysterious memo.
Jim: "It's time for our first quarter camaraderie event, so pack a swimsuit, a toothbrush, rubber-soled shoes, and a ski mask."

Pam: A ski mask and a swimsuit.

Jim: So that he can have us rob a bank, and then escape through the sewers.

Pam: And brush our teeth.
Michael: Yeah?
Stanley: Michael.

Michael: Stanley! Bo banley.

Stanley: I need to know...

Michael: Banana fana fo fanley.

Stanley: What we're doing.

Michael: Be my mo manley.

Stanley: You said bring a toothbrush.

Michael: Stanley.

Stanley: Is this an overnight?

Michael: Maybe. The suspense is just so exciting, isn't it?

Stanley: Should my wife tell her boss she's not coming in tomorrow?

Michael: Maybe, I don't know.

Stanley: Not maybe. Yes or no.

Michael: Well, no. But... okay, don't spoil it for everybody, all right? But we are going on a booze cruise on Lake Wallenpaupack.

Stanley: In January?

Michael: It's cheaper.
Michael: This is not just another party. This is a leadership training exercise. Right? I'm going to combine elements of fun and motivation and education into a single mind-blowing experience.
Michael: It is now time to unveil the destination of this year's retreat. We are going on a harbor cruise of Lake Wallenpaupack. It's a booze cruise!
Meredith: All right!

Ryan: I have a test for business school tomorrow night. Is it okay if I skip the cruise and study for that?

Michael: No. This is mandatory. But don't worry, you know what? You're gonna learn plenty. This is gonna turn your life around, Ryan.

Ryan: I'm already in business school.

Michael: Well, this...

Kelly: Wait, Michael?

Michael: Yeah?

Kelly: Why did you tell us to bring a bathing suit?

Michael: To throw you off the scent.

Kelly: Yeah, but I bought a bathing suit.

Michael: Well, just keep the tags on and you can return it.

Kelly: I took the tags off already.

Michael: Well, that's not my fault, okay? Just.. we're not going to pay for a bathing suit. Okay, I know what you're all thinking, "Who is this smart little cookie?" Her name is Brenda... something, and she is from corporate. And she is here, like you, to learn from what I have to say.
Michael: I am a great motivational speaker. I attended a Tony Robbins event by the airport last year, and... it wasn't the actual course. You have to pay for the actual course. But it talked about the actual course. And I've incorporated a lot of his ideas into my own course.
Michael: Leader... ship. The word "ship" is hidden inside the word "leadership," as its derivation. So if this office is, in fact, a ship, as its leader, I am the captain. But we're all in the same boat. Teamwork!
Oscar: Last year, Michael's theme was "Bowl over the Competition!" So guess where we went.
Michael: Now, on this ship that is the office, what is a sales department? Anyone?
Darryl: How about the sales department is the sails?

Michael: Yes, Darryl, the sales department makes sales. Good. Let me just explain. I see the sales department as the furnace.

Phyllis: A furnace?

Jim: Yeesh, how old is this ship?

Pam: How about the anchor?

Phyllis: What does the furnace do?

Michael: All right, let's not get hung up on the furnace. This just... it's the sales... I see the sales department down there. They're in the engine room, and they are shoveling coal into the furnace, right? I mean, who saw the movie Titanic? They were very important in the movie Titanic. Who saw it? Show of hands!

Jim: I'm not really sure what movie you're talking about. Are you sure you got the title right?

Michael: Titanic?

Pam: I think you're thinking of The Hunt for Red October.

Michael: No, I'm Leo DiCaprio! Come on!
Jim: Michael stands in the front of the boat and says that he's king of the world within the first hour, or I give you my next paycheck.
Phyllis: Michael, everyone in the engine room drowned.
Michael: No! Thank you, spoiler alert. You saw the movie, those of you who did. They're happy down there in the furnace room. And they're dirty and grimy and sweaty, and they're singing their ethnic songs, and... actually, that might be warehouse.

Darryl: What?

Michael: The... no, no. No, I didn't... okay. Well, okay, in a nutshell, what I'm saying is... leadership. We'll talk more about that on the boat. Ship.

Dwight: Aye aye, Captain.
Michael: [singing] A three-hour tour, a three-hour tour.
Michael: Pam, you are Mary Ann! We have the Professor and Ginger, welcome aboard. Angela, you are Mrs. Howell. Lovey. [to Kelly] Uh... the native. Sometimes they come from neighboring... [to Stanley] We have one of the Globetrotters, I am the Skipper, and Dwight, you will be Gilligan.
Dwight: Cool.

Captain Jack: Actually, I'm the Skipper. But you can be Gilligan.

Michael: I'd rather die. Hi, I am Michael Scott, I am the captain of this party.

Captain Jack: I am Captain Jack, I am captain of the ship. I'm also captain of anyone who sets foot on the ship. [to boarding passengers] Hi, welcome aboard.

Michael: Okay.
Michael: In an office, when you are ranking people, manager is higher than captain. On a boat, who knows? It's nebulose.
Michael: Hey, look! I'm king of the world!
Captain Jack: Okay, all right! Welcome aboard! I am your captain, Captain Jack.
Michael: And I am the regional manager of Dunder-Mifflin, Michael Scott. Welcome, welcome!

Captain Jack: Okay! So...

Michael: Okay! So...

Captain Jack: Please. The life preservers.

Michael: Right.

Captain Jack: They are located underneath the seats, all along the border of the boat.

Michael: But don't worry, you are not going to be needing life preservers tonight.

Captain Jack: Well, we might, okay? Please let me finish, okay? Thank you. So, the Coast Guard requires that I tell you where the safety exits are. On this ship, it's very easy. Anywhere over the side. [Dwight laughs loudly.] Not only am I your ship captain, I am also your party captain! Whoo! We're gonna get it going in just a few minutes here...

Michael: I'm your party captain too! And you are gonna put on your dancing shoes later on! So we are gonna...

Captain Jack: Okay, Michael, if you don't mind...

Michael: Rock it!

Captain Jack: Please, okay?

Michael: If the boat's a-rockin', don't come knockin'!

Captain Jack: Michael.

Michael: Yep.

Captain Jack: Your company's employees are not the only people on the boat tonight, okay?

Michael: We're all gonna have a good time tonight!

Captain Jack: Why don't you let me and my crew do our job. You just sit back and have a good time. All right?

Michael: Hm? Okay. Yep.
Katy: You guys, it's like we're in high school and we're at the cool table. Right?
Roy: Yeah.

Katy: Pam, were you a cheerleader?

Roy: No, she was totally Miss Artsy-Fartsy in high school. She wore the turtleneck and everything!

Katy: That's hilarious.

Jim: It's not hilarious, but...

Roy: Where did you go to school?

Katy: Bishop O'Hara.

Roy: Piss slop who cares-a? We played you! You... you really look familiar. Did you... you cheered for them, didn't you?

Jim: Um, no.

Katy: Yes, I did! [chanting] A-W-E-S-O-M-E! Awesome! Awesome is what we are! We're the football superstars! A-W-E-S-O-M-E!

Roy: I remember that! We crushed you like 42-10!
Michael: Having fun?
Brenda: Yeah. Everybody's really nice.

Michael: Good. Well, that is what Scranton is all about. Not like you New Yawkers.

Brenda: When are you going to start the presentation?

Michael: Well, we already sort of started it back at the office and on the dock with the Gilligan thing, so... right now, I was thinking. Yes. Okay, listen up all you Dunder-Mifflinites! I would like to talk to you all about life preservers. Now, one important life preserver in business is IT support.

Captain Jack: Not now, Mike, we're doing the limbo! That's right, partiers, it's time to limbo, limbo, limbo!

Michael: So, okay.

Dwight: Limbo, whoo!

Captain Jack: All right! I need a volunteer to come up here and hold my stick. Who's it gonna be?

Meredith: Me.

Captain Jack: Okay...

Dwight: Me! Me, me, me.

Captain Jack: Uh... usually it's a woman.

Dwight: I'm stronger.

Captain Jack: Hey, I got an idea! How would you like to steer the ship, Dwight?
Captain Jack: Keep us on a steady course. Keep a sharp eye out. I'm counting on you!
Dwight: I was the youngest pilot in Pan Am history. When I was four, the pilot let me ride in the cockpit and fly the plane with him. And I was four. And I was great. And I would have landed it, but my dad wanted us to go back to our seats.
Captain Jack: All right, all right, that was great! Now it's time for the dance contest!
Michael: But before that, I have to do my presentation.

Captain Jack: Nope! Dance contest!

Michael: All right, we'll have a motivational dance contest! Hit it! Yeah, okay, dancing! It is a primal art form used in ancient times to express yourself with the body and communicate!
Michael: Sometimes you have to take a break from being the kind of boss that's always trying to teach people things. Sometimes you have to just be the boss of dancing.
Dwight: [singing] What do you do with a drunken sailor? What do you do with a drunken sailor? What do you do with a drunken sailor early in the morning?
Angela: Hey, come inside and talk to me.

Dwight: I can't. Do you want us to run aground, woman?!
Darryl and Katy: [chanting] Snorkel sh*t! Snorkel sh*t!
Roy: Whoo! Who's next? Come on, Pam! Come on! Come on!

Pam: No, I'm not going to do that.

Roy: Come on!

Darryl: That's what I'm talking about!

Pam: Hey, why don't we find like a quieter place to hang out?

Roy: I've just gotta wait for Darryl to do his sh*t. Just a minute. Come on! [chanting] Darryl! Darryl!
Pam: It's getting kind of rowdy down there.
Jim: Yeah. [chanting] Darryl! Darryl! Darryl!

Pam: Sometimes I just don't get Roy.

Jim: Well...

Pam: I mean, I don't know. So... what's it like dating a cheerleader?

Jim: Oh, um... [A long silence.]

Pam: I'm cold.
Captain Jack: So, what's this presentation all about?
Michael: Ah! See, this is of general interest. It is about priorities and making decisions, using the boat as an analogy. What is important to you? If the boat is sinking, what do you save?

Captain Jack: Women and children.

Michael: No, no. Salesmen and profit centers.

Captain Jack: That's a stupid analogy.

Michael: Okay, well, obviously you don't know anything about leadership.

Captain Jack: Well, I was the captain of a PC-1 Cyclone Coastal Patrol Boat during Desert Storm.

Dwight: Wow. You should be the motivational speaker.

Michael: Okay.

Dwight: Yeah. He gives me real responsibility, Michael. Captain Jack delegates. He's let me steer the ship for the last hour.
Katy: I'd like to be engaged. How did you manage to pull that off?
Pam: Uh, I've been engaged for three years, and there's no end in sight. So... you don't wanna ask my advice.
Captain Jack: Suppose your office building's on fire. Jim, who would you save?
Jim: Um... let's see, uh... The customer. Because the customer is king.

Michael: Not what I was looking for, but a good thought.

Captain Jack: He's just sucking up!

Roy: When you were in the Navy, did you ever almost die?

Captain Jack: Oh yeah, oh yeah. And I wasn't thinking about some customer. I was thinking about my first wife. The day I got back on shore, I married her.
Jim: You know what? I would save the receptionist. I just wanted to clear that up.
Roy: Hello, everybody, could I have your attention for just a second? Could you listen to me for a second? We were up at the front, and we were talking about what's really important, and... Pam, I think enough is enough. I think we should set a date for our wedding. How about June 10th? Come on, let's do it! Come on, Pam!
Michael: I don't want to take credit for this, but Roy and I were just having a conversation about making commitments and making choices. Right? Did I motivate you?
Roy: No, it was Captain Jack.

Michael: Well... could have been either one of us, because we were pretty much saying the same thing. Congratulations. That is great!

Captain Jack: We gotta celebrate! Hey, I got an idea, I got an idea. I can marry you right now, as captain of the ship!

Michael: Yes! I can marry you as regional manager of Dunder-Mifflin!

Pam: No, no, I want my mom and dad to be there.

Michael: Then I'll give you away!

Pam: No, thank you.
Katy: Do you think that'll ever be us?
Jim: No.

Katy: What is wrong with you? Why did you even bring me here tonight?

Jim: I don't know. Let's break up.

Katy: Whoa. What?
Captain Jack: This is where Captain Jack drives the boat.
Meredith: Wow!
Dwight: Seasick? Captain Jack says you should look at the Moon.
Michael: Captain Jack is a fart face. I'm on medication.

Brenda: Really? What?

Michael: Vomicillin. Okay. All right. It's time to be boss. It's time to motivate. Let's blow some minds here. Okay, guys, guys, cool it. Everybody, Dunder-Mifflin Scranton employees, Brenda, I have some very, very urgent news I need to tell everybody right now. Listen up. The ship is sinking! Okay? We're going down, right now. Just wrap your heads around the reality of that. Shh, please! Everybody, it's my turn now, okay? Captain Jack is gone. In five minutes, this ship is going to be at the bottom of the lake! And there aren't enough spaces on the lifeboat! Who are we gonna save? Do we save sales? Do we save customer service? Do we save accounting? This is a business scenario. Right? It's a scary... it's a...

Captain Jack: Hey! Hey! What the hell is going on here?

Michael: It's a predicament, and it's something that each and every one of us has to think about.
Michael: I'm in the brig. See? The boat's not as corporate-friendly as advertised. What was the deal with the guy jumping overboard? What was... if he had just waited and heard what I had to say, he would be motivated right now and not all wet.
Michael: Is somebody there?
Jim: What happened to you?

Michael: Captain Jack has a problem with authority.

Jim: Oh, right, because you announced that his ship was sinking?

Michael: He just totally lost it. If you ask me, he caused the panic.

Jim: What a night.

Michael: Well, it's nice for you. Your friend got engaged.

Jim: She was always engaged.

Michael: Roy said the first one didn't count.

Jim: That's... great. You know, to tell the truth, I used to have a big thing for Pam, so...

Michael: Really? You're kidding me. You and Pam? Wow. I would have never have put you two together. You really hid it well. God! I usually have a radar for stuff like that. You know, I made out with Jan...

Jim: Yeah, I know.

Michael: Yeah? Yep. Well, Pam is cute.

Jim: Yeah. She's really funny, and she's warm. And she's just... well, anyway.

Michael: Well, if you like her so much, don't give up.

Jim: She's engaged.

Michael: BFD. Engaged ain't married.

Jim: Huh.

Michael: Never, ever, ever give up.
Dwight: Don't worry, Michael. I'm taking us to shore.
Michael: It's a fake wheel, dummy.
Oscar: ...Lord of the Rings trilogy, if you see it back to back, it's really long. But it's good.
Jim: [off camera] Yeah, that's right.
Pam: Dunder Mifflin, this is Pam.
Michael: Pam! It's Michael. Help me! I need help right now.

Pam: Michael, what's wrong?

Michael: I'm hurt, I have hurt myself. Oh my God!

Pam: Ok, wait wait wait wait...

Michael: Ungh, this is not looking good Pam!

Pam: Michael, do you need me to call an ambulance?!

Michael: No, I want you to pick me up.

Jim: What?

Pam: Ok...

Jim: What's going on?

Pam: Wait a second, I thought you said that you were hurt.

Michael: I am hurt. I hurt my foot.

Jim: I'm sorry? Pam.

Pam: [exasperated]

Jim: What is going on?

Michael: I want to come to work. But I need you to come and pick me up. [Jim lunges across Pam's desk and puts Michael on speakerphone]

Michael: OH GOD!

Jim: Hey, whoa, Michael...

Michael: Oh God!

Jim: It's, okay, it's Jim. Just say again, uh, really loudly what happened.

Michael: OK, buhhhh, I b*rned my foot very badly on my Foreman Grill and I now need someone to come and bring me into work.

Jim: You b*rned your foot on a Foreman Grill?
Michael: I enjoy having breakfast in bed. I like waking up to the smell of bacon, sue me. And since I don't have a butler, I have to do it myself. So, most nights before I go to bed, I will lay six strips of bacon out on my George Foreman Grill. Then I go to sleep. When I wake up, I plug in the grill, I go back to sleep again. Then I wake up to the smell of crackling bacon. It is delicious, it's good for me. It's the perfect way to start the day. Today I got up, I stepped onto the grill and it clamped down on my foot... that's it. I don't see what's so hard to believe about that.
Michael: Pam, could you come get me?!
Pam: Uh, I have to stay here and answer the phone.

Michael: Ok, could someone come and get me please, Ryan?

Phyllis: Michael, you should stay home and rest.

Michael: There's no toilet paper here. Could Ryan... tell Ryan to bring toilet paper. Could you tell 'em that?

Kevin: Can you hop?

Michael: I tried hopping, Kevin, and I bumped my elbow against the wall and now my elbow has a protruberance.

Michael: [panicked] No one wants to pick me up!?

Dwight: [silence, Dwight enters the office] What is going on? What is going on?

Pam: Michael, is, um, sick and he wants one of us to rescue him.

Michael: I'm not sick! I'm b*rned!

Dwight: I'm coming Michael!

Jim: Oh...

Dwight: I'm gonna save you!

Michael: Don't... is that Dwight? I do not want Dwight.

Dwight: Hold on Michael! I am coming! Wait there!

Michael: I don't want Dwight!

Pam: Michael, why don't you call your girlfriend?

Michael: I don't have a girlfriend.

Jim: But you said that you went out with her this weekend.

Michael: It was all made up. Just someone come, ok? Anyone. Anyone but Dwight.

Jim: [sounds of a car crash] What was that...

Pam: What was that?! [everyone runs to Michael's office window]

Jim: Oh!

Pam: Ohhhhhh!

Jim: He hit the pole!

Jim: It's broken right, he can't...

Pam: Oh my gosh.

Jim: Oh Dwight, Dwight, [Dwight pukes on his back windshield] Ohhhhhh!

Jim and Pam: Oh my God!

Pam: Is he ok?

Jim: He's still driving... Dwight, you forgot your bumper!

Michael: Hellooo? ... Please don't send Dwight!
Michael: Morning everyone. Don't freak out. I forbid anybody to freak out. Clearly, I have had a very serious accident, but I will recover, God willing. I just want to be treated normally today. Normal would actually be good, considering the trauma that I've been through.
Pam: You missed two big conference calls today, one with corporate.

Michael: Did you explain why?

Pam: No, I didn't mention that you cooked your foot.

Michael: b*rned my foot, Pam.
Michael: Please stop popping my cast. Thank you.
Jim: So, where are you shipping your foot?

Michael: Ha ha ha. So where are you shipping...

Dwight: Your foot?
Michael: Thank you. Pam, messages please?
Pam: You didn't have any.

Michael: Really, well, it, uh, seemed very important to you earlier that you needed to stay and...

Pam: And do my job?

Michael: No, your job is being my friend, Pam. OW! God!

Dwight: [holding mini-fan] It slipped.

Michael: What?

Dwight: Sorry.

Pam: It's just that before, you said that you didn't want any special treatment.

Michael: I don't want any special treatment, Pam. I just want you to treat me like you would some family member who's undergone some sort of serious physical trauma. I don't think that's too much to ask.

Pam: Do you want some aspirin, because you seem a little fussy.

Michael: No, I don't want some aspirin, yeah I'm a little fussy. Aspirin's not gonna do a damn thing. I'm sitting here with a bloody stump of a foot.

Dwight: Hey, Pam, I'm assistant regional manager, and I can take care of him. Part of my duties are to.

Michael: What? Part of your duties are to what?

Dwight: What?

Michael: You just said "part of your duties are to" something.

Dwight: No, I didn't.

Michael: Yes, you did. What is wrong with you?

Dwight: What is wrong with you?
Michael: Where is my cornbread?
Ryan: Here you go.

Michael: Thank you. Did you get all dark meat like I like?

Ryan: Yes. I ordered three full rotisserie chickens worth of all dark meat.

Michael: Where are the yams?

Ryan: They were out of yams. I got you creamed spinach.

Michael: Did you go to the one in Stroudsburg?

Ryan: Yes.

Michael: And they had no yams?

Ryan: They had no yams.

Michael: How strange. Because they always have yams.
Dwight: Aw, man, is that a Prism Duro-Sport?
Pam: You've seen one of these?

Dwight: Yeah, they're like an i-Pod only they're better 'cause they're chunkier and more solid.

Pam: Roy gave it to me for Christmas. I'm trying to figure out how to put songs on it.

Dwight: Oh, no no no. Don't go there. I know this Russian website where you can download songs for two cents a piece.

Pam: Really?

Dwight: Yeah, I'll write down the address for you. Only, the only thing is, is that all the songs are in Russian. ... Kidding!

Pam: Oh! Ha, haha.

Dwight: Why would they all be...? Ok, see you later, Pan.

Pam: Pan?
Michael: Pam... PAAAM!?
Pam: Oh, God.

Pam: [phone rings] What.

Michael: Come here please.

Pam: Tell me before I come there.

Michael: I want you to rub butter on my foot.

Pam: No.

Michael: Pam, please? I have Country Crock.

Pam: No.

Michael: Uh, ow. Ryan! ... Ryaaaaan ... RYYYYAN!
Dwight: These covers are totally indestructible.
Pam: Really?

Dwight: Yeah. Throw it. I promise it won't break. Chuck it. [Pam throws her mp3 player]

Dwight: Oh no, it's broken.

Pam: What?!

Dwight: No, it's fine. I told you it wouldn't break. You could throw it all day long.

Pam: That is so cool. Thanks Dwight!

Dwight: Yep.
Jim: So, I guess Pam and Dwight are friends now.
Pam: Oh God no, Dwight isn't my friend... Oh my God! Dwight's kind of my friend!
Michael: No, nope, no one is helping me out at all Mom. No, I'm not gonna call Jan. She'd just worry... drive down here and make a big thing... Who told you that? No, it was mutual. What is Pam doing chatting with you?
Dwight: Huh. Do you like candy?
Angela: It's alright.

Dwight: Cause you're sweeter than candy.

Angela: What is wrong with you? [Dwight pats Angela on the rear and runs away laughing]

Angela: Hey!
Toby: Wow, you just dive right into it.
Ryan: You know, around age twelve, I just started goin' for it.

Michael: [loud noise in bathroom] No! Guh! OW! Awww, help, help me!

Toby: What, what happened?

Michael: I fell off the toilet. I'm caught between the toilet and the wall.

Toby: What do you need?

Michael: Ugh, not you. Someone else. Get Pam.

Toby: I don't think Pam's gonna want to come into the men's room.

Michael: Get Ryan. He needs to lift me. [Ryan shakes his head] and he needs to clean me up a little bit. Bring a wet towel.

Toby: Ryan, is, uh, dead.

Michael: No, he's not.

Toby: Dead.

Michael: I just saw him.

Toby: No. Can't, can't you just get up yourself? I... You only grilled your foot.

Michael: Ugh, forget it. I'll just get up myself. No! Uh, aaaahhh! Ah! Oh God!
Jim: Do you think Dwight's bein' a little weird today?
Pam: No, he's actually been really nice and helpful.

Jim: And that isn't weird?

Pam: Wow...

Michael: Can I have everyone's attention please? Phyllis, Oscar, Ryan, who's supposed to be dead, can I ask you all a question? Do you all know what it's like to be disabled? Oscar?

Phyllis: Um, I had scoliosis as a girl.

Michael: No, never heard of it. No, a real disability, not a woman's trouble.

Creed: When I was a teenager, I was in an iron lung.

Michael: Wuh, how, how old are you? Look, the point is, I am the only one here who has a legitimate disability, although I'm sure Stanley has had his fair share of obstacles.

Stanley: I'm not disabled and neither are you.

Michael: Ok, [lifts up cooked foot] what does this look like to you Stanley?!

Stanley: Mailboxes, Etc.

Michael: Shuuut it, ok, well, well you know what, disabilities are not things to be laughed at or laughed about. You people are jerks. Imagine if you had left Stevie Wonder on the floor of that bathroom instead of me.

Phyllis: Oh, we wouldn't. We love Stevie Wonder.

Michael: [sigh] I b*rned my foot!!! Ok, twenty minutes, conference room, everybody's in there!

Dwight: [looking up at Creed] Dad?
Michael: While we are waiting for our special guest to arrive, I wanted you all to take a look at a few of the many, many disabled icons who have contributed so much to our society.
Jim: Quick question: uh, why is Tom Hanks on the wall?

Ryan: Twice.

Michael: Good question. Forrest Gump: mentally challenged, Philadelphia [points to a picture from Big][/b]: AIDS.

Kevin: I think that's from Big.

Michael: I don't think so, no.

Kelly: Yeah, he's dancing on a piano with Robert Loggia.

Michael: He grew into a man overnight. Rare disability, still works. [sigh] A crossword puzzle Stanley, seriously, are you learning nothing here?

Stanley: Uh hmmmm... .

Michael: What you mean uh hmmm... ?

Stanley: I mean I'm learning nothing.

Michael: Ok.

Billy Merchant: Michael Scott, I'm looking for Michael Scott.

Michael: Yes, right in here, come on in.

Billy Merchant: Great.

Michael: This, ladies and gentlemen, is our special guest.

Billy Merchant: Sorry I'm late. Someone parked in the handicapped parking space.

Billy Merchant: Hey everyone, I'm Billy Merchant, you may have seen me around here before, I'm the properties manager of this office park

Michael: You are so brave. You are so brave.

Billy Merchant: Thank you. Actually, I've been meaning to come by here for a long time...

Michael: But it's hard for you! Right? Because you're in a wheelchair.

Billy Merchant: No, I just have a lot of properties to manage.

Michael: Let me ask you something, how long does it take for you to do something simple, every day, like brush your teeth in the morning?

Billy Merchant: I don't know, like 30 seconds?

Michael: Oh my God, that's three times as long as it takes me.

Michael: How did you get in your wheelchair?

Billy Merchant: This morning? Just like every other morning, just climbed on in. [Everyone laughs]

Michael: Hey, hey, hey, not funny! Not funny.

Billy Merchant: Hey, hey, relax, just jokin around here.

Michael: Well, that's good, he still has a sense of humor.

Billy Merchant: Listen, I've actually used a chair since I was four years old. I don't really notice it anymore.

Michael: Well they notice it. Don't you? You notice it. It's the first thing you saw when he rolled in here, isn't it?
Jim: I want to clamp Michael's face in a George Foreman grill.
Billy Merchant: So, there are just a couple things I want to remind everybody of...
Michael: Ok...

Billy Merchant: First is parking. You can't block the freight entrance with your car, even if your blinkers are on. Does anybody have any questions? [to Dwight, whose arms is raised] Yes. Yeah? yes...

Pam: Dwight, you have your hand up.

Michael: Ignore him. You know what? We're not that different, you and I. When I clamped my foot into a non-stick...

Billy Merchant: You know what Michael?

Michael: Yeah...

Billy Merchant: Let me stop you right there.

Michael: Ok.

Billy Merchant: And leave.
Michael: Did you see Born on the Fourth of July? I was under the impression that Billy would be more like that guy.
Billy Merchant: What's wrong with that guy?
Jim: You mean today? He stepped on a George Foremen grill and he b*rned his foot.

Billy Merchant: No, not Michael. The moon-faced kid who crashed into the pole. He looks like he has a concussion.
Michael: [popping his bubble wrap cast] Hey!
Ryan: I found the pudding cups you wanted in a gas station in Carbondale!

Michael: You did it! Look at you, and with the plate and the napkin. Very nice. Thank you, Ryan.

Ryan: You are very welcome.

Michael: Did you get the yams?

Ryan: No, the gas station in Carbondale did not have fresh yams!

Michael: [sigh] Ok, I'll just have the pudding.

Ryan: You sure?

Michael: Yeh.

Ryan: Ok.
Michael: You know what? I feel better. Ryan brought me some chocolate pudding and his kindness healed my foot.
Michael: Yeah, baby, I am feelin' better. My body's literally healing itself. It is amazing what the human body is capable of when you have a powerful brain.
Ryan: I ground up four extra-strength aspirin and put them in Michael's pudding. I do the same thing with my dog to get him to take his heartworm medicine.
Michael: Uh, finally feel the blood coursing through my foot veins.
Dwight: [hits his head on his desk] Uh, ugh, ohhhh...

Jim: Uh, ok, I think we need to take him to the hospital because I'm pretty sure he has a concussion.

Michael: Oh, now you feel some compassion for him.

Angela: He needs to go right now, and you're his emergency contact. I think that you should go with him.

Michael: Why don't you go with him?

Angela: I, barely know him...

Dwight: I want Michael to take me...

Michael: I can't take you, I don't have my car and yours is all vomity.

Meredith: You can take my van!

Michael: Oh, ok, that's, great. No, I can't drive. Jim why don't you drive.

Jim: Fine.

Michael: We'll go. I'm still recovering. So let's just, Ryan, could you get my coat please.

Jim: Slowly, slowly. Let's just get to the elevator.

Dwight: Choo choo choo choo choo choo...

Jim: What are you doing? What, stop...

Dwight: Vietnam sounds.

Jim: [Dwight falls onto the couch] Stop, stop, stop, stop.

Dwight: Tired... [Jim grabs spray bottle from planter]

Jim: You can't lay down.

Dwight: Want to take a rake... .

Jim: Wake up. [sprays Dwight]

Dwight: Ahh!

Pam: Dwight, here, let me help you Dwight.

Jim: I'm just gonna get...

Dwight: Ok, Pam, thanks.

Pam: Get up, get up.

Dwight: You're the best.

Pam: Yeah.

Jim: Just keep him awake.

Dwight: It smells like chicken soup.

Pam: I know.

Dwight: I have to go to the hospital.

Pam: I know.

Dwight: Where we going?

Pam: I just want to say goodbye ok?

Dwight: I'll be back, I mean...

Pam: Yes, I know, but it's gonna be different.

Dwight: Why?

Pam: It's just hard to explain.

Dwight: Aw, Pam, you're adorable [taps her nose]

Pam: Oh my goodness!

Dwight: [giggles]

Pam: Come here.

Dwight: Oh, huggy hugs.
Michael: g*n!
Jim: You don't think you should sit in the back with Dwight?
Michael: The rules of g*n are very simple and very clear. The first person to shout "g*n" when you're within the sight of the car gets the front seat. That's how the game's played. There are no exceptions for someone with a concussion.
Michael: Oh, God, a mini-van. What is Meredith's problem?
Jim: Well, I think she has a kid.

Michael: Well, yeah she has one kid, no husband. She's not gonna find one driving this thing around.

Dwight: Where are we going?

Jim: Come on, get inside.

Dwight: Where are we going?

Jim: We're going to Chuck E. Cheese.

Michael: Chuck E. Cheese? Oh, God, I'm so sick of Chuck E. Cheese.

Jim: We're going to the hospital, Michael.

Michael: I know, just sayin'.
Michael: Dwight, what are you drinking?
Dwight: I found it under the seat.

Jim: Oh my God, Dwight, put that down.

Dwight: I'm thirsty.

Jim: Give the bottle to Michael [sprays Dwight]

Dwight: No!

Jim: Give the bottle to Michael!

Dwight: I'm thirsty!

Michael: Give it to me.

Dwight: No.

Michael: Dwight... [to Jim] You just keep your eyes on the road. [to Dwight] Give me the bottle or you're fired.

Dwight: You can't fire me, I don't work in this van!

Michael: Give it to me Dwight.

Dwight: No. [takes a drink]

Michael: Give me the bottle!!

Jim: [to Michael] Will you stop?

Michael: Gimme the bottle, Dwight!

Jim: Michael stop.

Dwight: [drinks] Mmmmm...

Michael: Just give it!

Jim: Michael stop. [sprays Michael, then Dwight]

Michael: Stop, stop it! Stop spraying! [Dwight whines] Gimme the bottle!

Jim: Stop [sprays Michael]

Dwight: My eyes!

Michael: Stop spraying me! Gimme the bottle!

Dwight: My eyes!
Michael: Dwight, what is your middle name.
Dwight: Danger.

Michael: [sigh] Something with a "K".

Jim: It's Kurt. Wow, I am so sad that I know that.

Michael: What do I write under "reason for visit"?

Jim: Concussion. [Michael scribbles something out] What did you write?

Michael: Nothing. I wrote "bringing someone to the hospital".

Jim: So you thought they meant your reason for coming to the hospital.

Michael: No... you know what Jim, this isn't about me anymore. I made a miraculous recovery, which is more than I can say for him. [Dwight falls towards Jim]

Jim: Come on Dwight. [sprays Dwight]

Dwight: Hi Michael!

Michael: Hi Dwight.
Dwight: Ahhh. Mweehaa
Michael: Doctor, what is more serious, a head injury or a foot injury?

Doctor: A head injury.

Michael: Well, you don't have all the information. The foot as been fairly severely b*rned and healed quickly, very quickly, actually like suspiciously quickly.

Doctor: [to Dwight] So, I'm ordering a CAT scan.

Dwight: What is that?

Michael: Look since you have the machine up and running, can I just stick my foot, we take a look?

Doctor: Well, for a burn, you really just need to look at the outside of the foot.

Michael: Ok, what kinda machine is that?

Doctor: Does the skin look red and swollen?

Dwight: That's what she said.

Michael: That's my joke, damnit Dwight.
Lab Tech: Ok, no electronics past this point. Camera, sound equipment...
Michael: It's ok, they're with me.

Lab Tech: No metal of any kind.

Michael: Alright, well, I guess this is where we leave you off.

Dwight: I don't want to do this.

Michael: Uh, well you should of thought of before you crashed your head on your way to pick me up. We'll, see you when you get out.

Dwight: Oh.

Michael: Fine. Fine.
Pam: Dunder Mifflin, this is Pam.
Jim: Dunder Mifflin, this is Jim.

Pam: Oh my God, what is going on, is Dwight ok?

Jim: Uh hmm, he should be fine, but, uh, they brought him in for a CAT scan.

Pam: I can't believe he's getting a CAT scan.

Jim: Michael went in there with him too. It's pretty sweet.

Pam: Really? Michael went in with him?

Jim: Uh huh.

Pam: Wow.

Jim: But they shouldn't be much longer now, so we'll be back soon.

Pam: Ok, that's uh, good news [Pam sees Angela eavesdropping] Uh, yeah, no I'll let you go.

Jim: Ok.

Pam: Ok. Bye.

Jim: Bye.
Pam: Hey, Oscar?
Oscar: What's up, Pam?

Pam: I just wanted to let you that Dwight's gonna be ok. The doctor said there's a really simple treatment for a concussion, so he'll probably even be back at work tomorrow.

Oscar: Ok...

Pam: I just, uh, thought you'd want to know that.
Lab Tech: Ok Mr. Schrute, inhale with me on three. One, two, uh Sir? [Michael tries to put his leg in the scanner] Stop that. Stop. Stop that.
Jim: Not much what's up with you?
Pam: Oh, I can not believe I fell for that. [laughing] Oh, my God.

Michael: Ah, ah, ah, what? What? Where's the funny? Give it to me.

Jim: Umm, is it me or does it smell like up-dog in here?

Michael: What's up-dog?

Jim: Nothin' much what's up with you?

Michael: Oh, oh, wow! I walked right into that. Oh, that's brilliant!
Michael: Hey, Stanley, is that jacket make of up-dog?
Stanley: I'm on the phone.
Michael: Mmm, what flavour coffee is that? Up-dog?
Ryan: What's that?

Michael: I don't know, nothin', what's up with you?

Ryan: Huh?

Michael: [low] No, damn it!
Kevin: What does that mean?
Michael: What does what mean?

Kevin: The thing you just said?

Michael: Just forget it.
Michael: Dwight! Hey is it me or does this place smell like up-dog?
Dwight: What's up-dog?

Michael: Gotcha! [laughing] Oh, God. [low] Crap! Nothin' how ya doing?

Dwight: Good. How are you doing?

Jim: [mouthing] So close.

Michael: [low] Damn it.
Michael: Today is spring cleaning day here at Dunder Mifflin. And yes I know its January. I am not an idiot. But, if you do your Spring cleaning in January; guess what you don't have to do in the spring? Anything. They say a cluttered desk means a cluttered mind. Well I say that an empty desk means a...
Dwight: Empty mind.

Michael: No, that's not... no, that's not what I was going to say.
Dwight: Meredith, men's room. Make sure you replace the urinal cakes. They're worn down. Kevin file drawers. Angela kitchen. Oscar dusting. Where is Oscar?
Angela: He's out sick.

Dwight: That's unacceptable.

Angela: I agree it's unacceptable. [longing look]

Kevin: Whhh... What are you guys doing?
Dwight: Michael.
Michael: Yes.

Dwight: Oscar is out sick.

Michael: On a Friday? [Dwight nods]
Dwight: Can I do some of the talking?
Michael: I will do all the talking.

Dwight: Ok, let him know that I'm here.

Oscar: Hello.

Michael: What difference does it make whether your here?

Oscar: Hello?

Michael: Hi, Oscar its Michael.

Dwight: And Dwight.

Michael: Yechh, yeah, um, heard you were under the weather?

Oscar: Yeah I think I came down with the flu.

Michael: Really? Oh, that is a shame. You know it's cleaning day here today? Could have used some of that famous Hispanic cleaning ethic.

Oscar: Yeah, I feel terrible about it.

Dwight: Ask him his symptoms. I'm on Web M.D.

Michael: What are your symptoms?

Oscar: I have the chills.

Michael: Umm, hmmm.

Oscar: I feel nauseous and my heads k*ll.

Dwight: Checks out.

Oscar: Michael is there anything you need from me? I'd like to go back to bed.

Michael: I need you to go back to bed. I need you to get better. See you Monday. Unless you're still sick. So have a great long weekend.

Oscar: I'll just be sleep--- [Michael hangs up the phone before Oscar can finish]

Dwight: Ok. First impressions?

Michael: He sounded sick.

Dwight: Which is exactly how you'd wanna sound like if you wanted someone to think you were sick.

Michael: That's exactly what I was thinking.

Dwight: Question? May I investigate?

Michael: Yeah. Drop what you're doing. Make this a priority. Because an office can't function efficiently unless people are at their desks doing their jobs.
Pam: I bought my veil.
Kelly: Oh my God! That is so exciting! Can I be a bridesmaid?

Pam: Ummm...

Kelly: Listen, you don't have to answer now. But how are you going to do your hair?

Pam: Ok. I was thinking about wearing it down. Kind of like, I don't know, like loose with big curls and...

Kelly: You'd look like an angel. I'm seriously going to cry.

Michael: Wowweee. Mikey likey. Why don't you wear your hair like that all the time. It's much sexier. [Pam puts hair back up] [Michael walks by Jim] Man, this must be t*rture for you.
Jim: Yeah. On the booze cruise I told Michael about some feelings I used to have for Pam. I had just broken up with Katy and had a couple drinks. And I confided in the world's worst confidant.
Jim: Hey Michael.
Michael: Hey Jim-bag.

Jim: Remember that thing I told you on the booze cruise about Pam? That... was... personal so if we can just keep that between you and me. That would be great.

Michael: Really?

Jim: Umm, hmm.

Michael: Who else knows?

Jim: Nobody.

Michael: Wow!
Michael: Jim and I are great friends. We hang out a ton, mostly at work. But, the fact that he told me his secret and no one else knows says everything about our friendship. And it is why, I intend on keeping that secret for as long as I possibly can.
Michael: My lips are sealed. [singing] My lips are sealed... Bangles.
Jim: Alright. Great. Thank you.

Michael: [singing] Can you hear me, they talk about us...
Dwight: Listen Temp. I am conducting a little investigation so I'm no longer going to be able to head up spring cleaning. Do you think you can handle it?
Ryan: Yeah, I think I can handle it.

Dwight: Do you think? Or do you know?

Ryan: I think.

Dwight: [low] Oh God, here.
Michael: Hey, whatcha gettin'?
Jim: I'm going with grape.

Michael: Ah, good stuff, good stuff. Did you see the game last night?

Jim: Which one?

Michael: Any of em? So, uh, what's the 411? Any news on the "P" situation?

Jim: I don't know what you mean.

Michael: P-A-M. P-A

Jim: Uh, uh, ok.

Michael: No it's okay, we're talking code.

Stanley: What is?

Michael: Listen Stanley. How long does it take you to pick out a soda?

Jim: I'm going to take off actually.

Michael: Alright, well, cool. [Michael walks by Jim] Still deciding?

Stanley: Hmm?

Michael: [Michael presses a button for Stanley] Peach iced tea. You're going to hate it.
Dwight: Hey Oscar how ya doin'? Dwight Schrute calling. Listen a little question for you, buddy. I called six minutes ago and no one answered. So I was wondering if you could explain. Oh, I see, so. Sounds like you're too sick to come into work but your well enough to go to the pharmacy.
Dwight: There are several different ways to tell if a perp is lying. The liar will avoid direct eye contact. The liar will cover part of his or her face with his hands, especially the mouth. The liar will perspire. Unfortunately I spoke to Oscar on the phone so none of this is useful.
Michael: It's Grrrrrrape! Soda.
Jim: Tony the tiger. You don't hear that much any more.

Michael: Not so much.

Dwight: Ok, what is going on here?

Michael: Nothing.

Dwight: Oh, really nothing? Fact: You are drinking grape soda. You never drink grape soda. Fact: You are talking to Jim. You never talk to Jim.

Michael: Fact: I love grape soda. I always have. Fact: Jim and I talk all the time. We tell each other secrets.

Dwight: Ok. So what is the secret Michael?

Jim: Um, I had asked Michael if I could head up the Oscar investigation and he said that only Dwight was capable of handling such sensitive material.

Dwight: Is that true?

Michael: Um, I don't know, yeah, yeah, yeah it is.

Dwight: Thank you Michael. I know your telling the truth.

Michael: Ok.

Dwight: I can tell. I won't let you down.

Michael: Good.

Jim: Thanks.

Michael: Whooo, nice. That was, that was slick. What are you doin' for lunch?

Jim: I don't know probably just gonna eat my ham and cheese sandwich in the break room.

Michael: Oh nonsense [lifts leg and puts it on Jim's desk], no way, no. Why don't, why don't I take you out to lunch? My treat.

Jim: No, that's alright, thank you though. It's, I, gotta do some cleaning, should probably stick around here.

Michael: Hey you know what we could do? We could spread out a blanket in the break room. Have a little picnic order some 'za. Talk about you know who.

Jim: Oh, ah, no but no. You know what let's go out. That was a good idea. Let's go out.

Michael: I know just he place.
Michael: [at Hooters] Oh man, you should order milk. Get it?
Michael: Why do I like Hooters? Well I will give you two reasons, the boobs and the hot wings.
Michael: Oh, here we go, here we go. Bogy at 3 o'clock. Hi.
Dana: Hey I'm Dana. Welcome to Hooters.

Michael: We're not worthy. We're not worthy. Hello Dana, I am Michael and this is Jim and we are brothers.

Jim: Nope we're not brothers.

Michael: I'm his boss actually. And I treat him well. I'm taking him out to lunch cause I can afford it and he can have whatever he wants.

Jim: Can I just have the ham and cheese sandwich, thanks.

Dana: And for you?

Michael: Tell me Dana, how is your chicken breast?

Dana: Oh, it's great. It's served with our world famous wing sauce.

Michael: Mmmm, sounds yummy. I will have a chicken breast hold the chicken. [Giggles]

Dana: Is that what you really want?

Michael: No, I'm gonna have the gourmet hot dog.

Dana: Great.
Dwight: Who took all the black ones?
Pam: That's a communal bowl.

Dwight: So, how did Oscar sound when he called in?

Pam: Sick, like lots of sniffling. I don't know.

Dwight: Sniffling how?

Pam: Umm. How many different ways are there to sniffle?

Dwight: Three.

Pam: Ok, it was the second one.

Dwight: Ok, good, thank you. That wasn't so hard now was it?

Pam: Nuh-uh.
Michael: What do you like best about Pam?
Jim: Uh, I really don't want to talk about it.

Michael: Is it her boobs, or...

Jim: Um, she's easy to talk to I guess and she's got a really good sense of humor.

Michael: Really?

Jim: Uh-huh.

Michael: Never get's any of my jokes.

Jim: What about you?

Michael: Her boobs, definitely.

Jim: Wow, that's not what I meant.

Dana: Here you go.

Michael: Oh, thank you.

Dana: And I understand we have a birthday today.

Michael: Ohhh happy birthday Jim!

Dana: Ready girls? Front side.

Hooter's Girls: You put your front side in; you put your front side out. You put your front side in and shake it all about. You do the hokey pokey and you turn yourself around. That's what it's all about. Whoo, hoo!

Jim: Thank you.

Michael: Woo! Yeah!

Jim: Thanks, thanks Dana.

Michael: Thank you very much.
Michael: Hilarious. Hey.
Pam: What did you guys talk about?

Jim: [Holds up Hooters t-shirt] Just you know politics, literature.

Pam: I hate you.
Dwight: Quick Oscar update. I have conducted interviews with everyone in the office.
Michael: Just go to his house and see if he's sick. I could have done this Investigation in like twenty minutes.

Dwight: Including prep time?

Michael: Just do it.
Ryan: If I had to I could clean out my desk in five seconds and nobody would ever know I had ever been here. And I'd forget too.
Michael: [Michael messes up hair to look like Jim's] Expenses.
Kevin: Michael is that a wig?

Michael: No. It's... I wear it like that sometimes. Is that a wig?

Kevin: No.

Angela: This is from Hooters.

Michael: Yeah, it's a business lunch.

Angela: Did Toby approve this?

Michael: No he did not. I don't need his permission.

Toby: You just got your corporate credit card back. Do you really want me to take it away again?
Michael: Uhhh it's ridiculous. They took my card away because I spent $80 bucks at a magic shop. What they don't understand is that I bought the stuff to impress potential clients. So business related, right?
Michael: I put a cigarette through a freakin' quarter. And you know what Toby? They almost bought from us.
Toby: I'm not processing this.

Michael: Look Jim needed a relaxing lunch, he has been depressed and it has been affecting his productivity. How is that not work related?

Toby: He seems fine to me.

Michael: You're not his friend, you don't know. He is in love with a girl he works with who's engaged. So just cut me some slack. Please?

Kelly: Pam?
Phylis: Angela who would you choose Jim or Roy?
Angela: It's nobody's business, Phyllis. Roy.
Kevin: Jim has got it bad for Pam.
Creed: Oh ho! Which one is Pam?

Kevin: Well she's the... Hey Michael so do you think Jim will try to break up the wedding?

Michael: You know what Kevin? Jim is a friend of mine, so the only people that this crush really concerns is Jim and Pam... and me.
Dwight: As a volunteer Sheriff's Deputy I have been doing surveillance for years. One time I suspected an ex-girlfriend of mine of cheating on me. So I tailed her for six straight nights. Turns out she was, with a couple of guys actually so... mystery solved.
Kelly: Jim, why didn't you tell me you had a crush on Pam?
Jim: Well the cats out of the bag. I used to have a crush on Pam and now I [hesitate] don't. Riveting.
Kevin: Nice... she is so hot.
Pam: Hey.
Jim: Hey.

Pam: Did you find anything good in your desk?

Jim: Ah, coupon for a free sandwich.

Pam: Score.

Jim: It expired in August, and my cell phone charger from two years ago.

Pam: Big day.

Jim: Big day.

Jim: Hey oh, listen, um, I told Michael on the booze cruise. It's so stupid. Um, I told Michael that I had had a crush on you when you first started here.

Pam: Oh.

Jim: Well I thought that, I figured you should hear it from me rather than, I mean you know Michael.

Pam: Right.

Jim: And seriously, it's totally not a big deal, ok? And when I found out you were engaged, I mean.

Pam: No, I know, like, I kind of like, I thought that maybe you did when I first started.

Jim: Oh you did?

Pam: No, I mean, just 'cause we like got along so well.

Jim: No, no, you saw through me, great.

Pam: So are you going to be like totally awkward around me now?

Jim: Oh yeah, yeah... hope that's okay.

Pam: Mmm, hmm.

Jim: And Pam it was like three years ago so I am totally over it.

Pam: Cool.

Jim: Ok.
Dwight: Stay low... This is it... There he is. He's been gone for at least two hours. Who is that? Come to Papa... Oh yes. Let's roll. I knew it! You are so busted. Ice skates, shopping bags? I think I know what's going on here. You weren't sick at all.
Gil: Who's this?

Dwight: This is Dwight Schrute. Who is this?

Gil: Gil.

Oscar: Are you going to tell Michael?

Dwight: How bout this. I don't tell Michael and in exchange you owe me one great big giant favor. Redeemable by me at a time and place of my choosing.
Dwight: Guess what I found out about Oscar tonight? He was lying about being sick. Should I have reported Oscar's malfeasance. Hmm, probably, but now I know something he doesn't want me to know. So I can use his malfeasance to establish leverage. Otherwise, it's just malfeasance for malfeasanceses-ses sake.
Jim: Hey.
Michael: I know, I know, I know.

Jim: Umm, what happened?

Michael: I, oh, just, um, I know I was trying to, expense reports. And then God, Toby, you know he just... I know. I'm just, I just hope that, I just hope that [starts to get choked up] this doesn't affect our friendship! Stupid, this is so stupid.

Jim: Hey, hey, wow, wow. Listen man it's, you know what. It's not a big deal.

Michael: Ok, I'm fine, no I know, I'm good, I'm good, it's just.

Jim: Look its one day, everything's gonna be alright. No big deal. You good?

Michael: Yeah I'm good.

Jim: Good.
Ryan: Creed did you organize the menu book?
Creed: Oh, I thought that was more on a volunteer basis.

Ryan: No, that was mandatory.

Creed: Oh, I thought it was a volunteer thing.
Pam: Hey, here's your schedule for next week. Are you okay?
Michael: Yeah I'm fine. Look, about you and Jim.

Pam: Oh, no, that's, you don't have to.

Michael: No, I feel it's my responsibility as your boss slash friend.

Pam: No, really, it's okay. I know that Jim had, like a crush on me when he first started. But that was a long time ago, so.

Michael: It wasn't that long ago. It was on the booze cruise.

Pam: Jim had a crush on me on the booze cruise or he told you about it on the booze cruise?

Michael: Yehhh, okay, shuuttt it Michael. I'm done. That's it. I'm out.
Jim: Ready?
Pam: Yep.
Michael: People are always coming to me. "Michael, I have a secret. Your the only one I trust." No thanks, because keeping a secret can only lead to trouble. Like I was watching Cinemax last weekend. This movie, Portrait of a... Prostitute something. Secrets of a Call... More Secrets of a Call Girl. And the lead character, Shila, is framed for m*rder. She goes on the run and winds up working at a bordello in Malibu. I don't, I don't want to live like that. I like it here. I don't want to be Shila, I like being Michael Scott.
Ryan: [catching Jim looking at him at Pam's desk] What?
Jim: Oh, nothing.
Jim: Pam's on vacation and she gets back tomorrow, so it'll be nice to see her. It'll be nice, and, uh, she set a date for the wedding with Roy. Uh... June. Summer. So, that'll be nice. And that's that.
Ryan: [again catching Jim looking at him] What?
Jim: Oh, nothing.
Ryan: Jim's been looking at me kind of a lot all week. I would be creeped out by it, but it's nothing compared to the way Michael looks at me.
Michael: Spamster!
Pam: Um, Pam plus Spam plus...?

Michael: Hamster.

Pam: Right.

Michael: Welcome back! How was your vacation?

Pam: It was great.

Michael: Yeah?

Pam: Mm-hm.

Michael: Did you get lucky? Oh! Boink!
Pam: Roy and I just got back from the Poconos. I get ten vacation days a year, and I try to hold off taking them for as long as possible, and this year I got to the third week in January.
Michael: I am Pam. Spicoli guy. Oh, God. Names, numbers. Okay. [walking into office] Whoa! God! Yuck, yuck. Yuck. Yuck!
Pam: What?

Michael: Wow! What happened in there?

Pam: I don't know.

Michael: There is stink in there, my God! What is... what is that?

Pam: [looking at pile on Michel's carpet] Oh... I don't know.

Michael: Is it a bird?

Pam: No, I don't think it's a bird.

Michael: Oh, God! How could that happen? How could... right in the middle of the carpet.

Kevin: What's goin' on?

Michael: Um, somebody vomited right in the middle of the carpet in my office.

Kevin: [taking a look] I don't think that's vomit.

Michael: Check it out.

Kevin: Me?

Michael: Check it out. Don't be a wuss, just get... no, I'm not holding your coffee.

Kevin: Oh, that's ridiculous.

Michael: What is it?

Kevin: Michael. [tapping on door]

Michael: What is it? No, just tell me what it is.

Kevin: [pounding on door] Michael, I ... I ... I gotta get outta here. I can't hold my breath that long.

Pam: Open the door up!
Kevin: It smelled terrible.
Pam and others: [after going in to check out the smell] Phew. Oh! No, mm-mm. [leaving quickly]
Michael: I cannot believe a pipe burst and left that in there.

Toby: That's no burst pipe.

Michael: How do you know that? What is it, then?

Creed: Hi guys. Somebody makin' soup?
Michael: [as cleaning lady with mask leaves] Here she comes. All cleaned? Great. [walks into office]
Dwight: [coughing] It's still stinky.

Michael: That is worse.

Dwight: She probably scrubbed it into the fibers of the carpet. Total permeation.
Michael: [while in his reeking office] I am a big Fear Factor fan. I'm a big fan of anything Joe Rogan does, actually, so this is sort of like my audition tape. Um... [clearing throat] I can't stand it [gets up to leave], I can't stay in here another second. No!
Jim: Hey! Welcome back!
Pam: Thanks!

Jim: So, how was the resort? Did you ski a lot?

Pam: A little.

Jim: Good! What's goin' on here?
Jim: What? I did not do that. That sounds disgusting.
Ryan: [barely stifling laughter] It wasn't me. Um... it wasn't me. [regaining composure] It was not me.
Jim: [smelling the stink] Oh. Wow.
Pam: [giggles at Roy]

Michael: [sitting at Jim's desk] Hey Jim. I thought that we would be desk buddies while they changed my carpet.

Jim: That might be a little difficult with the one computer.

Michael: Oh... It's ...

Jim: But there's definitely a desk open in the back.

Michael: [reluctantly] Yeah ...

Jim: ...which I guess I'll be taking.

Michael: No, no, no! Seriously, I don't mind sharing.

Jim: No, no, no, seriously, I'll be in the back.
Jim: Hey, Kelly.
Kelly: Are you moving back here?

Jim: Um, just for the day while Michael's at my desk.

Kelly: Because Toby used to sit there, but he had to move over there because of an allergy.

Jim: Allergy to... the desk?

Kelly: [shaking head] Weird.
Michael: [putting his feet on desk] Yeah, yeah, yeah. Old bullpen.
Dwight: [putting his feet on desk] Ha ha ha... the old bullpen.

Michael: Don't ape me.

Dwight: Okay.

Michael: This is great.

Dwight: Yeah!

Michael: The pressures of my office are insane.

Dwight: [agreeing] Mm.

Michael: I just... you couldn't understand, but man, you guys have it so easy out here, you know? I used to sit right here.

Dwight: No way!

Michael: Yeah.

Dwight: And who had your office?
Michael: Ed Truck. [exclaiming is disgust] Ed Truck was the manager before me. Horrible. He hated fun. It was like, "Oh, Ed Truck is walking toward us. Stop having fun. Start pretending to do work." What a jerk. He's... You know what? I swore to myself that if I ever got to walk around the room as manager, people would laugh when they saw me coming and would applaud as I walked away.
Kelly: [to Jim] I'm serious. My closet doors will not shut. I mean, it only takes so long to measure to make sure that clothes will hang up because aren't all hangers like that big? So I don't understand why the closet engineer didn't think of that. So now I'm doing this new thing where I just leave piles of clothes on the floor and then I walk around the piles to get an outfit...
Michael: You know who used to sit at that desk?
Dwight: That guy Miles who quit to form his own company?

Michael: Mm-mm. Todd Packer.

Dwight: No!

Michael: Yeah.

Dwight: I thought he was out on the road.

Michael: He was, but, uh... that desk was empty. He'd come in and sit there sometimes.

Dwight: Ah.
Michael: When I was in training, many years ago... not so long ago... I worked side-by-side with a fellow named Todd Packer, and together we rocked the office [picture behind Michael falls]. Packer and I once spent the whole day with our pants off, and when people noticed, we convinced them that they were crazy.
Michael: I will gladly pay you Tuesday for a hamburger today.
Stanley: [on phone] Excuse me one second, please. [to Michael] What is it that you need right now that you can't wait until I'm off the phone with a customer?

Michael: Oh, a customer, well, sound the alarm. [laughs] Okay.
Michael: Another time, Packer held this guy's head in the toilet for like a minute. Guy had no sense of humor about it. Probably why he wasn't hired.
Creed: [after Michael punches him in the arm] What did you hit me for?
Michael: Charley horse!

Creed: What?

Michael: Charley horse!

Creed: You shouldn't have hit me, Michael.

Michael: Oh, okay. Gah.
Michael: Once, as a joke, Packer banged every chick in the office. [giggles] It was hysterical.
Kelly: [to Jim] Beyonce, pink the color, Pink the person, hot dogs, basically anything that is awesome. Snow cones...
Ryan: Hey Jim, Michael wanted me to ask you how to raise your desk chair.

Jim: It's the lever on the side.

Ryan: That's what I told him. Thanks. [leaves]

Kelly: Oh my God, he is so cute! Would you talk to him for me and see if he likes me?

Jim: No, I don't think I can...

Kelly: Oh, please Jim? Please, please, Jim. Please, please, please? He's so cute. I like him so much. And I would do it, but I'm too shy. Please, Jim, please, please, please, please, Jim. Please, please, please...
Michael: [whispering] Dwight.
Dwight: [whispering] Michael.

Michael: Let's send up Accounting.

Dwight: What?

Michael: Old fashioned raid. Sales on Accounting. Yeah. Follow my lead.
Michael: Hey guys.
Oscar: Hey, Michael.

Michael: Ahem. What's up?

Oscar: Hey, Dwight.

Michael and Dwight: [as they throw accountants' files and supplies around] Ahhhh! Whoo hoo! Come on, come on, come on, come on! Sales rules!

Dwight: Yeah! [laughing]

Michael: Yeah! Oh ho ho [laughing]

Dwight: Should we help 'em pick up their stuff?

Michael: No, no, no, no. We don't do that. We don't do that.

Dwight: Okay.

Michael: Watch out, Pam. You're next!

Pam: You're gonna throw my things on the ground?

Michael: Maybe!
Oscar: What happened in Michael's office was wrong. I understand it [chuckles], it makes sense [regains composure] But it... it was still wrong.
Michael: Why would somebody ruin a perfectly good carpet? I don't know. It could be done out of hate. It could be done out of love. It could be completely neutral. Maybe somebody hates the cleaning lady. And, well, she doesn't do a very good job, obviously, because my office still reeks like you would not believe. I hate her.
Michael: You know what? I am beginning to think that what happened to my carpet was an act of terrorism against the office. The only thing that makes any sense.
Dwight: [on phone] Hello, am I the 107th caller? [hangs up, dials again] Hello, Rock 107. Am I the 107th caller? [hangs up and dials again] Hell , Rock 107. Am I the 107th caller? [hangs up and begins to dial again] I'm totally gonna win us that box set.
Michael: Stop.

Dwight: Jethro Tull...

Michael: Stop it. [Dwight hangs up] Stop. It. [Dwight beings to dial] Don't. Don't.

Dwight: I need to make a sales call. Please?

Michael: All right.

Dwight: [on phone, whispering] Am I the 107th caller?
Pam: [to Roy in Jim's earshot] ...back so soon.
Roy: We can go back in, like, a couple of weeks maybe.

Pam: Yeah, right.

Roy: Okay, maybe another month, like, maybe for, like President's Day or something.

Pam: Yeah, that's right. We could do a three-day weekend. I wonder if I could, like, call in sick on the Friday. Then I get a four-day weekend.
Kelly: [to unseen co-worker] But it's so weird to fall asleep. And I just hate it. 'Cause I try to go to bed at, like, 9:30.
Pam: [to Roy as Jim escapes into bathroom] Are you kidding?
Roy: No.
Michael: Hi, guys.
Angela: We haven't finished getting things in order from your last visit.

Michael: I'm just walking around.

Angela: Were you?

Michael: Well, yeah.

Oscar: It's just that we're really swamped over here, Michael.

Michael: Oh, and I'm not? Why would you say that? Because I'm having fun? You guys just are workin' for the weekend, aren't you? I'm workin' for the week. Sales team, listen to me. This is what we're gonna do. I'm gonna up the ante a little bit literally. Right here, I'm gonna put a crisp one hundred dollar bill. The person with the most sales at the end of the day gets to keep the cash. Sound good?

Dwight: Yeah!

Michael: [counting cash] Seventy, eighty, one, two three. Eighty-three dollars. Still a lotta money and I'm going to ... [moves money after seeing workmen walk by] ... I'm gonna leave it right over here where everybody can see it. I will be taking Jim's clients today because he is not here and out of sight, out of the contest. Let's see who winds up with the cash, shall we?

Phyllis: You're gonna compete against us?

Michael: Oh, it is on, Phyllis, it is so on!

Dwight: It is so on!

Michael: God, this is gonna be fun.

Dwight: Michael is gonna wipe the floor with us!
Michael: [on phone] So you have 40 boxes going out, and I will deliver those personally in a Sebring. Very good, nice doing business with you. Thank you. [hangs up] Yes! [chuckles] Oh, yeah! Read it and weep. Oh! Oh, look at that! [puts post-it on Phyllis' forehead] Look at me, Phyllis! Oh, what is that? That's my sale! [humming then dancing victoriously]
Darryl: [walking by with new carpet] What... What's that? Whatcha doing?

Michael: [stops dance] Nothing.

Roy: [laughing] I think he's dancing.

Michael: No. Just ...

Darryl: That was definitely not dancing.

Michael: You know what, guys? It's none of your concern. It was official business, so just...

Darryl: Paper business.

Michael: Yeah, paper business. Is this done?

Roy: Nope.

Michael: Extreme Home Makeover puts together a house in an hour. If you were on that crew, you would be fired like that. [snaps]
Pam: Somebody did something bad to Michael's carpet. Maybe that's all we need to know.
Creed: [to Oscar] Who do you think did it?
Oscar: Are you kidding? I thought it was you.

Creed: Really? I thought you. [both laugh in Michael's earshot]
Michael: This was no act of God. A person did this. A person who works in this office. Maybe all of them.
Michael: You know what? Today is not a good day for a sales contest. We're... we're not... we're not doin' this today.
Pam: That doesn't seem fair.

Michael: You wanna talk about fair? Does anyone need to smell my old carpet? You explain to me how that was fair, and I'll explain to you how this is fair. Plus I just... I think that picking today was sort of taking advantage.

Dwight: But you're the one who picked today.

Michael: I am a victim of a hate crime. Stanley knows what I'm talkin' about.

Stanley: That's not what a hate crime is.

Michael: Well I hated it! A lot! Okay, I... you know what? If the guilty person would just come forward and take their punishment, we'd be done . [no one comes forward] Very well. Then you are all punished.

Pam: What's our punishment?

Michael: You're all on a time out. Just sit there quietly. [phone rings, Phyllis reaches to answer] No. NO! [phone continues to ring]
Jim: Hey!
Ryan: What's up?

Jim: Nothing much. Let me ask you something. It's actually little awkward.

Ryan: What?

Jim: What do you think of Kelly?

Ryan: I don't know. Depends if you like a little junk in ... [notices camera] Umm... She's really cool.

Jim: Are you interested in her?

Ryan: Yeah, totally.

Jim: Really?

Ryan: Did she say something?

Jim: She said lots of things.

Ryan: Do you know if she's looking for a long-term thing or if she'd be cool just hangin' out?

Jim: I have no idea.

Ryan: Can you find out?

Jim: Yeah. Sure.
Kelly: [to Jim] Oh, long-term, definitely. Fall in love, have babies, spend every second together... but don't tell him that, okay? Just tell him I'm, like, up for anything. I mean, I'm not a slut, but who knows?
Michael: Do you remember Ed Truck?
Creed: Sure. He hired me. How's he doing?

Michael: How would I know?

Creed: I thought you might.

Michael: My biggest fear is turning into him.

Creed: Michael, you should have much bigger fears than that.

Michael: [sighs] I wasn't talking literally, Creed. Yeah, being buried alive would be worse. Happy? Why am I talking to you?
Michael: [meeting Ed Truck in parking lot] Ed? Hi. Thanks for meeting me. Must be kinda neat comin' back.
Ed: Yeah. Should we go upstairs?

Michael: Uh, well, honestly Ed, I really don't wanna be up there right now.

Ed: So, what's the problem with my pension?

Michael: Oh, no, no, no. You're good. It was clerical. You're good. Um, well, somebody did something in my office, and I now think that they did it on purpose and it was directed at me.

Ed: Well, what was done?

Michael: I didn't get a good look at... it, but it smells horrible.

Ed: Yeah, somebody once did that in my office.

Michael: Really?

Ed: Yeah.

Michael: Well, that figures. So how did you deal with people not liking you?

Ed: You can't expect to be friends with everybody.

Michael: Well... s-sure I can.

Ed: No. They'll always think of you as a boss first.

Michael: Not necessarily. You can love a boss like you do a father.

Ed: I'm not sure that ever happens.

Michael: Well, okay. Different management styles.

Ed: Why can't your workers be your workers, family be your family, your friends be your friends?
Michael: Last week I would have given a kidney to anyone in this office. I would have reached right into my stomach and pulled it out for them. But now, no. I don't have the relationship with these people that I thought I did. I hope they ask, so they can hear me say, "Uh... no. I only give my organs to my real friends. Go get yourself a monkey kidney."
Jim: [on phone] Hey, Brenda. This is, uh, Jim Halpert from the boat. And I got your number from the corporate directory and, well, I was assuming that you probably gave it to them because you wanted me to ask you out, right? Um, so gimme a call back. You can get my number from said directory, um, or just check your e-mail 'cause I just sent you one. Yikes. Uh... give me a call back, I hope. I'll talk to you later. Bye.
Kelly: You just asked a girl out on the phone!

Jim: Yep.
Michael: [on phone] Yes.
Todd Packer: Hello, yes. I'm looking for a gay nerd named Michael Scott.

Michael: Who is this? How did you get this number?

Packer: Your mom, you gay nerd!

Michael: Oh my God. Packer. Packster. Whacky Pack. How you doin'?

Todd: Hey, did you get that package I left for you?

Michael: Uh... no. Did anybody see a package here today? No. How big was it?

Packer: It was pretty big.

Michael: Really?

Packer: Yeah.

Michael: Did you see a big package? Where did you leave it?

Packer: Left it in the middle of your office.

Michael: Really? Guys, did you see a big package in my office?

Roy: You mean the thing?

Packer: [laughs uproariously]

Michael: Are you kidding me? Oh!

Packer: Special delivery!

Michael: That was Packer! Oh, you're... you are dead. You are dead, my friend! That is hilar... Oh, God! Of course it was you.

Packer: Sit on the throne, Michael.

Michael: Oh. [laughs and claps] Yeah, yeah! Oh my God. It was Packer!
Michael: It takes an advanced sense of humor. I don't expect everybody to understand. It was done out of love, just like I thought. It's ah... God, these people are so... these are good people. We have fun. [giggles] We just have fun! Oh, I'm just so sorry that I threw the thing out.
Jim's voicemail: You have seven unheard messages.
Pam: [voicemail message for Jim] Hey, Jim. It's Pam. I keep looking up to say something to you and then Michael's there and it's horrible. Anyway, I'm bored. Come back!
Pam: [voicemail message for Jim] Hey, guess what? I moved my computer so I can't see Michael's head. It's working. I think I can have a career as a very specific type of decorator.
Pam: [voicemail message for Jim] Sudoku. Level moderate. 18 minutes. Suck on that, Halpert.
Pam: [voicemail message for Jim] I'll transfer you. Dunder Mifflin, this is Pam. Hold, please. Dunder Mifflin, this is ... okay, sorry. Michael was standing at my desk, and I needed to be busy or who knows what would've happened, so thank you.
Pam: [voicemail message for Jim] Hey, what's that word we made up when you have a thing stuck in your shoe? Anyway, I have a thing stuck in my shoe.
Pam: [voicemail message for Jim] Hey, I have a chance to sneak out of here early, and I'm not messing this up, so I'll see you tomorrow.
Pam: [voicemail message for Jim] Calling from my cell phone. I don't know if you guys figured out who did that to Michael's carpet yet, but I have a theory that involves an inter-departmental conspiracy. Everybody in the office. We need to talk.
Jan: So, I'm happy to be here. It's very nice to see all of you. You're all looking well.
Pam: Today's a 'women in the workplace' thing. Jan's coming in from Corporate to talk to all the women about... um... I don't really know what. But Michael's not allowed in. She said that about five times.
Jan: Women today, though we have the same options as men, we often face a very different set of obstacles in getting there. So...
Michael: [knocks] Hey, what's going on?

Jan: Michael... I thought we agreed you wouldn't be here.

Michael: Yeah... I... You know what... I... I... I just thought about it. I just have a few things I want to say.

Jan: What are you doing?

Michael: Hold... Just hear me out. What is more important than Quality? E-Quality. Now studies show that today's woman, the Ally McBeal woman, as I call her, is at a crossroads...

Jan: Michael.

Michael: No, just uh... you have come a long way, baby. But I just... just want to keep it within reason.

Jan: Michael.

Michael: They did this up in Albany...

Jan: You are not allowed in this session.

Michael: And they ended up turning the break room into a lactation room which is disgusting so...

Jan: Now you're really not allowed in this session.

Michael: Well, I'm their boss, so I feel like...

Jan: I'm your boss.

Michael: [stands up] Anybody want any coffee or...

Jan: We're fine, Michael. We just need you to leave, please.
Michael: Women in the workplace... yeah, translation "I have been banned from my own conference room so that Jan can talk in secret to all the girls." Oh! Sorry. 'Women of the workplace.' About what? I don't know. Clothes. Me. Eeegkh!
Jan: Ladies, I am so, so sorry. Can we start again? We were on such a roll. I... I... really apologize.
Pam: Jan.

Jan: Yes, Pam?

Pam: Michael's still at the door.

Jan: Michael!
Jan: [in the background] So one obstacle is how assertiveness is perceived differently in men and women. Men who are assertive will be admired. They're called... anyone?
Dwight: It's a terrible idea.
Jim: What is?

Dwight: Them in there all together. If they stay in there too long they're going to get on the same cycle. Wreak havoc on our plumbing.
Michael: Everyone. Guys. Circle up, please. Come on over. Bring your chairs. Toby, come on over. You're a guy... too... sort of. Let's do this!
Michael: [in the background] Well, first of all, I, uh, just want to warm up a little bit. Let's just clap.
Michael: Let's just clap. Ready? [clapping] Yeah! Yeah!
Dwight: Yeah!

Michael: That's what I'm talking about!

Jan: I don't know what you're doing here, Michael,

Michael: Just having a little 'guys in the workplace' thing.

Jan: ... but it's very destructive.

Michael: Why can't boys play with dolls? Why does society force us to use urinals when sitting down is far more comfortable?

Jan: Can you please do this somewhere else, Michael?

Michael: We have nowhere else Jan. This...

Dwight: We could do it in the warehouse.

Jan: Dwight, excellent idea. Go to the warehouse.

Michael: OK, OK, Fine. Yeah, actually, perfect. Perfect. You know what? There's another side to this place, gentleman. And I know we all love our cushy jobs and our fun, exciting office. But do you realize that underneath us, there's another world. The warehouse world. A world that is teeming with sweat and dirt and life. Life. The bowels of the office. These guys are down there, they are real men doing real man's work. We are going to learn how a warehouse works.
Michael: Oh, I think it's going to work out great. Because managing the warehouse is a very important part of my job. And I haven't been there in months.
Dwight: Remember on Lost when they met the Others?
Jan: I'm so sorry about that... um... so where were we? Pam, are you okay taking notes then?
Pam: Mmmhmm.

Jan: Please? Thank you. very much.
Michael: So let's meet the warehouse! Let's get some shots. Pan around there. This is Darryl, one of our warehouse staff. Darryl, what is your biggest fear?
Darryl: My biggest fear is that someone will distract us from getting all the shipments out on time.

Michael: You know, Darryl is actually the Foreman here and not Roy, which is cool. There's Roy riding the big rig. So Roy is actually going to be marrying Pam sometime this summer. And... uh, she's our receptionist. Sort of a Brangelina thing.

Roy: Why?

Michael: Brangelina is the Brad Pitt and Angelina... Roy...

Roy: I don't understand.

Michael: Roy and Pam. It's a Ram. It's a Ram thing.

Kevin: [talking to Jim] I bet Roy heard about you liking Pam. I bet he'll try to b*at you up.

Jim: Thanks for the head's up, Kev.

Kevin: I've got your back if he does. But try to stay out of it.
Michael: [points at math equasion on chalkboard] Uhhuhhuh. Just in case there's someone down here who shouldn't be. A little "Good Will Hunting" situation. All right. Troops. This is an important day. Big day. Now you may look around and see two groups here. White collar. Blue collar. But I don't see it that way. And you know why not? Because I am collar-blind.
Jan: Why don't we go around the table and all say something that we know we're good at. I will start. I am good at public speaking.
Meredith: Hi. I'm Meredith and I'm an alch... good at supplier relations.

Jan: Great. Phyllis?

Phyllis: I'm good at computer stuff, emails, spreadsheets, all that.

Angela: [disbelieving] Really?

Phyllis: I don't know. I thought that I wasn't going to be asked that...

Jan: No. Okay. Stop. Go on...

Angela: I've seen some of your spreadsheets.

Phyllis: Really? I thought they were pretty...
Pam: I don't know how I fit in with these women. Here. Or with Jan. Um... I mean we get along great. Fine. Um... I guess the person I have the most in common with is...
Roy: Jim... Halpert. Hey uh, I, uh, you know heard there's a rumor going around about you used to have a crush on Pam.
Jim: Oh, no, no. No.

Roy: No, it's cool, because I know you're a good guy. And I know that that crush ended a long time ago, so... you know. We're cool, right?

Jim: Yeah. Nope. Yeah. Definitely.

Roy: You know, it's great with me cause that way, glad she has a friend at work she can get through the day with. She's not all bap bap bap bap when she gets home.

Jim: Yeah. I like talking to her too.

Roy: So, we're cool, right?

Jim: Yes. Yeah.

Roy: All right.

Jim: Yep. Cool, man.

Roy: Sweet.

Kevin: [blows a sigh of relief]
Darryl: Hey, Mike, look. How bout we go upstairs, too. You know learn how the office works. We can all switch places today.
Michael: Oh... well... okay... yeah, you know what? I don't think... You.. You're... My job sucks compared to this. I don't think you'd like it up there.

Darryl: The experience...

Michael: Guys! Want to start unloading the truck?
Dwight: Okay. Let's go. Step up.
Michael: Check this out! Look at that! Look at that [squeezes blow-up doll] bwup-bwa! [talks in girly voice] Hello! How are... [regular voice] Oh! Kay. That is great. That is good stuff.
Meredith: In five years, I'd like to be... five years sober.
Jan: That is an excellent goal.

Meredith: Four and a half.

Kelly: I'll tell you one thing. I am not going to be one of those women schlepping her kids around in a minivan.

Jan: Great! Uh-huh?

Kelly: I want an SUV... with three rows of seats.

Women: [general murmuring of agreement]
Jan: Well, I'll be honest. One of the goals of these women seminars is to feel out if there's any standouts. Women who could be a valuable addition to our Corporate life.
Dwight: Michael wants us to bond so we need topics for conversation.
Jim: Ponies.

Dwight: No.

Ryan: How about rainbows?

Dwight: No.

Jim: Flowers.
Darryl: It's dangerous, Michael. Come on, get off this.
Michael: Hey, you're going to, going to hurt yourself.

Darryl: Mike.

Michael: Stand clear.

Darryl: Mike. Get off of the lift. Please. Come on now.

Michael: I'm fine, I'm fine. I'm fine.

Darryl: Look, would ya... look.

Michael: Oh, oh, oh! We'll get somebody to clean that up.

Darryl: We're the ones that got to clean that up!

Lonny: Dammit, Michael!

Michael: We ought to have this thing serviced.
Michael: So! Guy's gripe session. Here we are. Now, we definitely live in different worlds but we have a lot in common. We even like the same girls, some of us. That's going to happen, you know. We're guys, so...
Madge: Hey, do you want me to go?

Michael: No, why? Why would I... ? You could...

Madge: I'll go.

Michael: Stay or...
Phyllis: ...and a big walk-in closet.
Meredith: Oh, that's part of my dream too.

Kelly: Oh, me too.

Jan: Great, great. And Pam, what about you? What is your dream?

Pam: Well... I always dreamed of a house with a terrace upstairs. Plant flowers on it... stuff like that. Since I was a girl. Um... More seriously though, a husband that I love... Roy. And I love to draw. And I... I did a little in college and I'd still love to do something where I could work with art or graphic design in some way.

Phyllis: She's real good.

Pam: Thanks.

Jan: You know the company is offering a design training program in New York.

Pam: Well... I have a job right now, so I can't really take time off...

Jan: Well, it's only on weekends and then a few weeks in New York, but I'm sure that I could ask Corporate to help you out.

Pam: Well... it's just that the weekends aren't good because, um...

Jan: There are always a million reasons not to do something.
Michael: Let's start with the Warehouse. What bothers you as guys, you know?
Darryl: My priority is safety.

Michael: OK.

Darryl: So it really bothers me when somebody comes in here speeding around on a lift, playing with it like a toy. It kind of gets under my skin.

Michael: OK, Yeah. Yeah. Yeah shhh...

Darryl: Uh uh uh. Don't shush me.

Michael: I... That was just...

Darryl: That bothers me too.

Michael: I was breathing.

Roy: Pam shushes me. It drives me crazy.

Michael: I hate shushing. You know, that's the thing! What the... ok... what is our beef as human men.

Lonny: You know that's a good question, Hasselhoff. What bugs us?

Michael: OK. Alright. Good. Guys ragging on each other. That's what guys do... and we love it.
Jan: All right. Let's talk about clothing.
Phyllis: I'm excited about today. [whispers] I love girl talk.
Jan: Dress for the job you want, not the job you have. All right. You can use your clothing to send a message about your ambitions by wearing clothes that reflect what you aspire to be.
Angela: I'm not gaining anything from this seminar. I'm a professional woman. The head of accounting. I'm in the healthiest relationship of my life. I just think it's insulting that Jan thinks we need this. And, apparently, judging from her outfit, Jan aspires to be a whore.
Roy: I hate it when girls insist on taking them out to new restaurants every weekend night and then they're like "When are we going to go on a date-date?"
Guy: I hate that too! [general clapping and agreement]

Darryl: I hate that too.

Kevin: That sucks so much.

Guy: It totally sucks.

Dwight: Yeah and then they make you drive them to Church the next morning. Like "Gas ain't free!"

Lonny: Yeah, on our salaries, man, what do they expect? You know to take us out every weekend? You know what I mean? We're not millionaires.

Michael: I feel you.

Darryl: No, you don't. You don't feel us. How can you? You know what?

Dwight: Not literally.

Darryl: You say we're the same, but we get compensated very differently.

Michael: Yes.

Darryl: We work the same hours as you and you just said we work a lot harder

Michael: Ah, you do. So...

Darryl: But we get paid a lot less.

Dwight: Word.

Roy: Like next to no benefits.

Michael: I know. God! What is that?

Roy: Exactly.

Michael: It blows. It blows, man. Gah...

Darryl: You know this would not happen if we had a union.

Roy: That's what I'm talking about.

Michael: No. Whoa, whoa. Yeah.

Roy: Absolutely.

Darryl: That's what we need.

Guy: You know you're right.

Darryl: Man, see... That's what I've been sayin', man. We need to do this finally.

Michael: You know what? Is that necessary? Because you already sorta have a union... of guys.

Darryl: It's more than necessary, Mike. We need this. Roy? You still have that card from the Dockworker's Union?

Roy: In my truck.

Michael: Dockworker's?

Darryl: Man, hook you up.

Men: [generalized clapping]

Guy: Come on, man.

Michael: Yeah. You know what? I think the problem is the chicks.

Darryl: Union! Union, yeah.

Michael: The problem is the chicks. And you gotta blame them.

Darryl: Are you with us Mike?

Michael: Yeah-es.

Darryl: Welcome to the warehouse.

Group chant: Michael, Michael, Michael, Mi...
Jan: Another issue is inequality of pay between men and women. I'm sure that all of you have felt that before...
Michael: [knocks] This is important. Ladies, take a breather. Jan, I uh wanna... Can I help you? Um... I wanted to say that the guys downstairs are thinking about forming a union. And they have some good points...

Jan: What? A union! What...

Michael: Don't get hysterical.

Jan: I'm not...
Michael: Part of my job is knowing how to talk to women.
Michael: Let's... be... rational... here. What are the pros? What are the cons?
Jan: The cons are that everyone will lose their job. Michael. Everyone. Office, Warehouse. What do you think... the... pros... are... here?

Michael: Don't talk to me that way please. Just... they're going to want to hear this from you.

Jan: You got yourself into this Michael, so you get yourself out.

Michael: But we're bonding down there!

Jan: That's too bad.

Michael: I mean I just don't want to have to tell them something they're not going to want to hear.

Jan: I don't want to...

Michael: Ok. Come on Jan. After all we've been through...

Jan: Michael! Michael! Michael!

Michael: We have a history...

Jan: Michael.

Michael: ...between us.

Jan: Don't say another word.

Michael: I won't

Jan: Get yourself down stairs.

Michael: I'm just saying we have something... Ok. Whatever.
Ryan: You know what... we could get this done a lot quicker if we formed a type of assembly line.
Stanley: This here is a run-out-the-clock situation. Just like upstairs.
Jan: Sports metaphors are one of the ways women feel left out of the language of the office. Now, I know this might sound silly but a ... many women ask to go over it. So... Fumble means...
Phyllis: Mistake.

Meredith: Slip.

Jan: Right. Par for the course is a golf term. It means right on track. Below par means worse. Wait... that should mean better, that doesn't make sense.

Kelly: What about second base? Like if Michael said that he got to second base with you? Does that mean you like closed a deal?

Jan: Excuse me?

Kelly: I mean that's a baseball term, right?

Jan: I don't know what Michael was talking about. I don't know.
Kelly: [in the background] ...and you went to Chili's and he got to second base with you.
Jan: [in the background] Kelly, I don't know what Michael's talking about.

Kelly: [in the background] He told everybody so I just want to know is that a baseball term...

Pam: Hey.

Jim: Hey!

Pam: How's it going down there?

Jim: It's a complete... well, actually it's exactly what you'd expect, so... How are the girls?

Pam: Good. We watched a video about our changing bodies.

Jim: Did you really?

Pam: No. [laughs]

Jim: Oh.

Pam: Almost.

Jim: Good.

Pam: Um... but hey? Something kind of cool. There's this internship in graphic design that Jan was telling us about. She made it sound, like, really great.

Jim: Nice. Well, what's it all about?

Pam: Um...

Jim: I think you should do it. That's great!

Pam: It's really cool.
Michael: Cold front coming into the Warehouse. Uh oh! Better put on your ski boots! Woohoohoowoo. Waaaah! Happy New Year, Darryl! Hey,Darryl. You ever done this?
Angela: Are you married?
Jan: I'm divorced.

Phyllis: That must have been hard.

Jan: It was. Yes.

Kelly: You were probably feeling really depressed and sad and that's why you did that thing with Michael.

Jan: I think you should all spend a little more time thinking about your careers and less time on personal stuff.

Phyllis: Mmmm, I think we're all okay with the balance we've struck.

Angela: At least you don't have kids. You have no kids, right? Thank God.

Jan: Okay. Let's take five. I think we can all use five.

Kelly: How can someone so beautiful be so sad?
Michael: Hi.
Jan: Did you take care of the situation?

Michael: Yuh, yuh, yes! I... I have essentially...

Jan: Excuse me.

Michael: I have essentially. Yes. I've taken some...

Jan: Excuse me. I've been told there's been some interest in forming a Union and that Michael supported it. Obviously he's not a friend of yours because he didn't tell you the facts. So let me. If there is even a whiff of unionizing in this branch, I can guarantee you the branch will be shut down like that [snaps her fingers]. They unionized in Pittsfield and we all know what happened in Pittsfield. It will cost each of you a fortune in legal fees and union dues and that will be nothing compared to the cost of losing your jobs. So I would think long and hard before sacrificing your savings and your futures just to send a message. If you have any further questions you can direct them to... to Michael.
Pam: Dreams are just that. They're dreams. They help get you through the day. Like the thing about the terrace. It's nice but... um... I don't know. It was just something I read in this book when I was twelve. The girl in the book has a terrace outside of her bedroom and she planted flowers on it and I just loved that. Just always kind of stuck with me.
Jim: So you're not doing it.
Pam: How did you know?

Jim: Why not?

Pam: Just like no big reason. Just a bunch of little reasons.

Jim: Come on.

Pam: Roy's right. There's no guarantee it's going to lead to anything anyway.

Jim: Roy said that.

Pam: What? You have something you want to say?

Jim: You got to take a chance on something sometime, Pam. I mean, do you want to be a receptionist here, always?

Pam: Oh, excuse me! I'm fine with my choices!

Jim: You are?

Pam: Yeah.
Pam: It's impractical. I'm not going to try to get a house like that. Um... they don't even make houses like that in Scranton. So, I'm never going to... .
Michael: I'm just going to put this over there.
Darryl: This is not a good idea right here.

Michael: You did uh... okay.
Michael: Pizza. Great equalizer. Rich people love pizza. Poor people love pizza. White people love pizza. Black people love pizza. Do black people like pizza?
Michael: Hey. Um... look guys, I'm sorry. Sometimes Jan can be such a bitch.
All the Men: Generalized mumbling agreement. Yeah.

Michael: Hey, watch it, watch it. We have a relationship.
Michael: Thank you to our hosts.
Darryl: Hey Michael. This ain't over.
Michael: Ahhh! Excellent.
Michael: Is it good to be back. Yeah. I mean I love the guy stuff but to run an office you need men and women. You know why? Because you need to have that crazy sexual tension to keep things interesting.
Pam: Dunder Mifflin. This is Pam. Uh... hold, please.
Pam: I really like Valentine's Day in this office. It's kinda like grade school. Everybody gives out little presents and stuff. Like last year, Jim gave me this card, with Dwight's head on it, it was horrifying and funny and...
Pam: [Delivery man enters with a bouquet of red roses. Pam stands up to look at card.] Phyllis.
Delivery man: Would you sign here? [Phyllis gets up from desk and walks over.]
Pam: Roy and I are saving for the wedding, so I made him promise not to get me anything too big.
Meredith: "Happy Valentine's Day darling. Love Bob Vance, Vance Refrigeration."
Phyllis: Isn't he sweet?

Meredith: Yeah. Wow.
Michael: Alright Dwight, as you know I am heading to New York today. [Dwight holds up passport.] Doing a presentation on the branch to the new CFO.
Dwight: And you want me to come with you.

Michael: Nope. The opposite of that.

Dwight: I will stay here and run things on this end.

Michael: Ok, good.

Dwight: Question. Will you be seeing Jan when you're in New York?

Michael: I probably will, why do you ask?

Dwight: Well... It's Valentine's Day, and you guys, you know...

Michael: Yeah.

Dwight: Screwed.

Michael: What is your problem?
Michael: This is a business trip. I would have to be a raving lunatic to try to talk to Jan about what happened between us. Her words, not mine. She sent me an email this morning. But, it is Valentine's Day. It's New York. City of Love.
Michael: Hey, Pam. You heart N.Y., right? You want me to pick you up anything?
Pam: That's OK.

Michael: Alright.

Oscar: The best present would be, you do a good job in front of the new CFO.

Michael: Dude, I'm gonna nail it. Me in New York? Oh, I own that city. Fuggedaboudit! See ya!
Michael: Well here we go. On our way to New York. New York, New York. City so nice they named it twice. Manhattan is the other name.
Jim: So I broke up with Katy and haven't been dating anybody else, so this year I don't have to worry about Valentine's Day. It's gonna be good. I invited a couple of friends over. We're gonna play some cards and I'll end up winning a lotta money. Because, they're idiots. It's gonna be great.
Dwight: What's this? What is this?
Jim: I dunno, it's on your desk.

Dwight: Yeah, but who put it here? And for what purpose?

Jim: It was there when I sat down.

Dwight: [opens box and reads card] Happy Valentine's Day. [pulls out bobble head] It's me. I'm the bobble head. Yes! [Angela smirks in background] Ahh!
Michael: The meeting isn't 'til three, but I always like to come to New York little bit early and hit some of my favorite hunts, like right here, is my favorite New York pizza joint. And I'm gonna go get me a New York slice. [Michael walks toward Sbarro.]
Jim: Hey Kelly. What's up?
Kelly: Nothing. Oh except, oh my God Jim. Last night, Ryan and I totally, finally hooked up. It was awesome.

Jim: OH, that's great. I'm really happy for [starts to walk away]

Kelly: And it was so funny 'cause we were at this bar with his friends and I was sitting next to him the whole night and he wasn't making a move, so in my head I was like "Ryan, what's taking you so long?" And then he kissed me. And I didn't know what to say.

Jim: Wow.

Kelly: So I said, "Ryan, what took you so long?" And I just said that to him, can you believe that?

Jim: Wow.

Kelly: Oh my God, Jim, is that embarrassing? I'm embarrassed.

Jim: No, don't be.

Kelly: Oh, thank God, because I was nervous, Jim, you will not believe.

Jim: I bet.

Kelly: So nervous, but now -- now I have a boyfriend.

Jim: Alright. [Kelly squeals]
Ryan: [anguished] I hooked up with her on February 13th.
Michael: Here it is, heart of New York City, Times Square. Named for the good times you have when you're in it. Most people when they come to New York, they go straight to the Empire State Building, that's pretty touristy. I come here. Great places to eat. [points] We have Bubba Gump Shrimp, Red Lobster down there. Ya know. This is, this is the heart of civilization, right here.
Kevin: Woah. [Delivery man with flowers]
Pam: Guess what?

Phyllis: Really, Oh, they're from Bob again.

Pam: That's great. [Meredith scowls]
Michael: Everybody takes the subway in New York. It's fast, it's efficient, gets you there on time. It's a way to [turns and rushes back up stairs] Okay, there's a guy pooping in a cardboard box down there.
Michael: This is the world famous Rockefeller Center. Founded, of course by Theodore Rockefeller. This is a skating rink and I think the Rangers practice there sometimes and it's, that's Tina Fey [points]. That's Tina Fey from Saturday Night Live. Hello? Hello, hi? [walks over] OH, I'm sorry, I thought you were [Conan O'Brien walks in front of Michael], OK, I thought that was. She, she looked a lot like Tina Fey. [to camera] Hello, hello, I thought that was Tina Fey, but it wasn't. So... Are you serious? He was here? When, when I was talking to the fake Tina Fey? Come on! And are you, argh.
Dwight: Hello Angela. Did you hear, somebody rocked the house and got me the best present I've ever gotten.
Angela: Really? I wouldn't know anything about that, but I'm glad you enjoyed it.

Dwight: Oh I did. I did.

Angela: I didn't get anything for Valentine's Day.

Dwight: Oh, I bet you will before the day is over.

Angela: Really? Well, I hope I do.
Michael: I would love to live in New York someday. It's a big dream of mine. Work for corporate, with Jan. It'd be awesome. Go to Broadway shows, eat hot dogs. Scranton is great, but New York, is like Scranton on acid, no on speed, no on steroids. [Michael sees it's the end of a street.] OK, umm, I think, that's either the Hudson or the East, so we're back, should be back this way. There's a lotta pressure on me right now. It's like Michael Jordan, in the NBA finals. Or, like Stormin' Norman Schwarzkopf, and this presentation is desert storm and as soon as it's over, we will not have to deal with those Iraqis anymore. Let's do it.
Jim: [on phone] Nah that's alright. Spend money on her, instead of giving it to us. That's fine. No, I didn't even have a seat for you anyway. Yeah, hahaha, alright man, have a good night. Bye.
Kevin: Woah, woah [Delivery man with oversized bear]
Delivery man: Phyllis Lapin.

Pam: OH, Holy God!

Delivery man: It's from Bob.

Kevin: Man, that thing's bigger than I am.

Delivery man: No, it's not.

Kevin: Oh zip it.
Michael: There they are. What's up? Hey hey.
Craig: Hey.

Josh: Michael Scott. [sticks out hand for handshake]

Michael: Josh Porter, high five. [They high five] Bam.

Josh: You know Dan Gore from Buffalo.

Michael: Yeah, how ya doing? Nobody needs to introduce this guy. Craiggers. [bump fists]

Craig: What's up buddy?

Michael: You have been kicked out of every strip club in Albany, is that true?

Craig: Guilty, yeah.

Michael: So what's going on? What I miss?

Josh: Not much, they're uh, I guess running late upstairs, so we're just waiting for the presentations.

Michael: Cool. Good, good, good. Give us some time to catch up, and... [awkward silence]
Dwight: Pam. Hi, How ya doing? Good. Listen, uh may I speak with you... privately?
Pam: You can't fire me, Dwight, just 'cause Michael's not here.

Dwight: No, Pam, Just. Just, [tilts head away, towards another room]
Pam: You need to get something for your girlfriend.
Dwight: [same time as Pam] Girlfriend. Yes, and the reason I didn't get anything for this particular person - who shall remain nameless - is that she's not really the kind of person you'd think would be into Valentine's Day. She's kind of...

Pam: Tightly wound?

Dwight: (smirking) Exactly.

Pam: Ok, well, sometimes the gift is really about the gesture, you know, like what it means, instead of what it is.

Dwight: You mean, like a ham?

Pam: No, not like a ham. It's about doing something, so that the person knows that you really care about her.

Dwight: Ok, I get it.

Pam: That you remember her.

Dwight: Ok, shut up. I know exactly what to do. [gets up and leaves]
Josh: What about you, Craig, you lose anyone?
Craig: Oh man, Jan, called me in September and said "You gotta fire four people," and I was just like, "What?" Ya know?

Josh: Did you?

Craig: No, I just ignored her. She's the worse.

Josh: She is our boss.

Craig: She ain't my boss dude. I don't work for that bitch.

Michael: Ay, Kay. Come on, you know, that's not. Cool it.

Craig: What? You like Jan? How can you like Jan?

Michael: Maybe because she's my girlfriend. [starts retracting statement] Was, or not my girlfriend. She's... we hooked up and...

Josh: You hooked up with Jan?

Michael: You know, months ago, just once, It's, just stupid. Just forget it,

Josh: Yeah, let's change the subject.

Michael: Yeah, yeah.
Kelly: I don't know what he's thinking, but I would just be so psyched if we just dated forever.
Jim: Take it slow. 'Cause it seems like a lot of the time things like that need... [Ryan walks in]

Ryan: Soda.

Kelly: [to Ryan] Cool. Hey, so... do you want to... do something tonight? Or...

Jim: [under breath] Oh, no, not while I'm here.

Kelly: I mean, I know it's Valentine's Day, or whatever, but there's totally no pressure at all, of any kind. What so ever. So...

Ryan: I can't tonight. I have plans with my friends.

Kelly: OK, That's cool. I completely understand.

Ryan: Cool. Cool. OK.
Jan: Josh Porter, Stamford.
David: David [to Josh and shakes hands]

Josh: Nice to meet you.

Jan: And Michael Scott, Scranton.

David: Nice to meet you.

Michael: Ditto. [to Jan] How are you Jan?

Jan: Fine Michael. Thank you.

David: OK. So we are in the process of doing a complete review of the company's financial strengths. All I'd like to do today is to...
Jan: Nervous, no I'm not nervous. Well, I guess I'd be lying if I didn't say I was a little nervous. Umm, the new CFO is judging me on this too, and well, it is Michael, so. Yeah, I'm very nervous.
Josh: So with the twelve new local accounts, we had a total of four percent organic growth, which was just above our pre-year targets.
David: Thanks very much.

Josh: Thank you.

David: OK, Michael.

Michael: What is a business? Is it a collection of numbers and sales reports? Sure. But as you know, David and Jan, it is much more. [plays video on screen. David and Jan have confused looks on their faces.]

Michael: [video dialogue for "The Faces of Scranton"] Life moves a little slower in Scranton, Pennsylvania. And that's the way we like it. Because at Dunder Mifflin Scranton, we're not just in the paper business, we're in the people business. Let's meet some of the folks that make the Scranton branch so special. [video shows Stanley at desk] This is Stanley Hudson, one of our talented salesman. An African-American father of two, Stanley's dedication is no doubt one of the hallmark's of the foundation of the business we're hoping to build our bases on.
Michael: Yeah, I sh*t a bunch of footage around the office, edited it together on my Mac. I was thinking of entering it into some festivals. Probably won't. You know, not what this is about.
Michael: [video dialogue] And finally, Pam Beesly. Look at her. Look how cute. Not bad at all. As the receptionist, Pam is truly the gateway to our world. Well, I hope this gave you a little taste of what life is like here at Dunder Mifflin, Scranton. What it's like to walk a mile in Oscar's shoes. Or try on Phyllis' pants. Maybe even one of Angela's famous brownies. And you'll know, that you're home. [video says, "Great Scott!"]
Michael: Questions?
David: Wow. OK, OK, thank you Michael, that was great.
Michael: Yes, thank you.

David: But, for right now what, I would really like to know about is the branch's performance, so do you have that information as well?

Michael: Yes, absolutely David. Get that for you. I umm... [hands over report]
Delivery man: [with flowers] Can you sign?
Pam: Yeah.

Pam: [delivering plant to Oscar's desk] Oscar.

Angela: Nothing for me?

Pam: [walks away] Join the club.

Kevin: Whose it from? [to Oscar]

Oscar: My mom. [puts card in pocket]
Kelly: It's frustrating, because we'd be so perfect together.
Jim: You know what? Here's the deal, Kelly. It would be really nice if he was in to you, right? It'd be great, but he isn't.

Kelly: Yeah, it would be so great if he was.

Jim: Well, he's not, though. So you just gotta suck it up. You just gotta move on. Try to have some fun. Come to my poker game tonight.

Kelly: Okay, cool. Is it okay if I invite Ryan? [Jim leaves]
Dan: And that about does it, thank you.
Jan: OK. [looks toward Craig] Craig,

Craig: Yeah. Here's the deal. I did not understand this was supposed to be a full on... like report or whatnot.

Jan: Um, I'm sorry, what did you think financial presentation meant?

Craig: I was under the impression this was, more of like... a meet and greet type deal.

David: So, does that mean you don't have the numbers on your branch?

Craig: That is correct, yes.

Jan: Craig, you realize that we're trying to decide whether if drastic steps need to be taken?

Craig: Look, I'm sorry, I didn't know.

Jan: Well, the point is, is that doesn't exactly bode well for your branch.

Craig: Oh man, you know what? Michael made that stupid movie, he doesn't get into any trouble? Maybe I should have slept with you, too. [David looks at Jan, who glares at Michael.]
Michael: Oh, ok. Alright.
Jan: NO, NO I'm not, I'm not, I just... I just don't know what to do anymore, Michael. I mean, we're all gonna get fired.

Michael: No you're not.

Jan: Yeah, Michael - the CFO thinks that we slept together. Do you understand, people get fired for much less? And I just [scratches head] can't believe that you told everybody and we didn't even sleep together.

Michael: Technically, we fell asleep in the same bed. So...

Jan: Oh, God. Michael. It was months ago. It was once, It's over. Do you understand?

Michael: Yes. I'm sorry. I will fix this. I'll talk to him. I'll talk to David.

Jan: Surely, you cannot be serious?

Michael: I am serious. And don't call me Shirley. Airplane.
Dwight: Women are like wolves. If you want a wolf, you have to trap it. You have to snare it. And then you have to tame it. Keep it happy. Care for it. Feed it. Lovingly, the way an animal deserves to be loved. And my animal deserves a lot of loving.
Roy: Hey babe.
Pam: Hey.

Roy: You almost ready to go?

Pam: I guess, yeah.

Roy: What's wrong?

Pam: Nothing, it's just I had to sit here all day, while Phyllis got like an entire garden delivered to her.

Roy: What, you're mad at me?

Pam: I mean, I know that we said no big gifts, but I was kind of hoping you'd get me something for Valentine's Day.

Roy: Well, Valentine's Day isn't over. Let's get you home and you are gonna get the best sex of you life.
David: You understand this is a very serious situation.
Michael: No no no no no, yes I, OK, well, alright, here's the deal. It's my fault. This is, this is totally on me. Before you guys came in, I was talking to the guys. We were all chatting and I made a joke, a really dumb joke and Craig the idiot took it seriously. [Jan looks at Michael]

David: You made a joke?

Michael: I did, it was stupid. And Craig, you saw him, he's not the sharpest tool in the shed. Although he is a tool. [David grins]

David: Well I don't need to explain to you that even a joke about sexual relations with your boss...

Michael: I know. It was borderline at best and... And Jan is a fantastic executive and has all the integrity in the world and um, I'm really sorry. It will never happen again.

Jan: Uh, that's fine. Let's just forget it.

David: Good. [Michael leaves office]
Pam: Heading out?
Jim: Yeah. Alright, Beesly, Hey, Happy Valentine's Day.

Pam: Bye. [Jim leaves]

Phyllis: Goodnight Pam. [Leaves with oversized bear on back.]

Pam: Night Phyllis.
Jan: Oh, Michael. Thank you again for that, really. It was very nice.
Michael: Oh, no big deal. Really. Sorry again.

Jan: Oh, no, it's OK. [Puts hand in way of elevator door to stop from closing.] So, uh, Happy valentine's Day.

Michael: Yeah, Happy Valentine's Day. [Jan turns and then kisses Michael. Michael looks and sees camera, Jan turns and sees camera, too. Elevator door closes.]

Jan: Oh.
Michael: Oy vey... schmear. [Points at Fiddler on the Roof playing at Minskoff Theatre and does a dance.]
Michael: Let's think this through. If we ask Corporate for that then...
Dwight: They are either going to say yes... or no.

Michael: Could go either way. We don't know what they are going to say.

Dwight: Think it through.

Michael: Have to think it through. Because if they say no...

Jim: Can we not?

Michael: No! Yes, we have to! You know why? Because I don't like to be cooped up in that office! In that box! All day long. [Michael starts playing with a football in the office] Heisman! Because I need to think. Okay, Jim? Oh, Kevin, oh! [laughs] Nice catch. Mmmm, mmm, mmm,mmm. Os-car! Intercepted.

Jim: Still want that.

Michael: Give it to me. Phyllis, give me the ball. Ok, give me the ball. Give me, you guys... Creed give me the ball! Right now give it to me.

Creed: Ryan!

Dwight: Fumble! Yaaah!

Michael: Hey, Dwight.

Dwight: Hut! Hut! Hut! Hike!

Michael: You all right Ryan?

Dwight: Ryan.

Ryan: Yeah.

Michael: Pam!
Dwight: Ooh. They're having a sale on TiVo. Maybe I should get a TiVo. Oh. DVD Burner! Maybe I should get one of those. You are so lucky, Jim. You are so lucky you don't have this problem. What was the 9th place prize again? A loaf of bread?
Jim: Cugino's pizza.

Dwight: Oh, great. Tasty, terrific pizza. Hmm. Question: Do their pizzas play DVDs?
Jim: Dwight was the top salesman of the year at our company. He wins a little prize money and gets honored at some convention. It is literally the highest possible honor that a Northeastern Pennsylvania-Based Mid-size Paper Company Regional Salesman can attain, so...
Jim: What did I do to deserve this?
Pam: Are you sad that Dwight b*at you?

Jim: No.

Pam: Are you going to cry, Jim? Do you need a tissue?

Phyllis: Hey, I heard you got a wedding dress. Do you have pictures?

Pam: Oh! I... uh... yeah. Um... I'll uh show them to you later.

Phyllis: Oh.

Jim: Oh, I should get back. Talk to you guys later.

Pam: Ok, cool.
Pam: I have a ton of stuff to do for the wedding. And I have to do it in the office. And that can be kind of awkward. Um... just because people can get all weird about wedding stuff. Then... I just... I don't want to offend... Angela... or someone.
Michael: That's what she said!
Dwight: Ha! I don't get it.

Michael: Grapes. Seductive. So you ready for the big speech this afternoon?

Dwight: Well, it's not really a big speech. You still coming right?

Michael: Oh! Abso-fruit-ly. Fruit. Grapes. Nailed the joke. Matter of time. Um... And yes, it is a big speech. Biggest of your life.
Michael: Speaker at the Sales Convention. Been there, done that. Went there again, did it again. Two years in a row. Consecutive. I just... I miss the feeling of knowing that you did a good job because someone gives you proof of it. Sir, you're awesome! Here's a plaque. What, a whole year has gone by and you need more proof? Here's a certificate. They stopped making plaques that year.
Dwight: What if I give a really long, extended Thank You. For instance, "Thank you, Mr. Blank. Thank you very, very, very..."
Michael: That would look terrible. These are mostly salesmen and salesmen expect to be entertained and you are the main act.
Dwight: When I was in the sixth grade, I was a finalist in our school Spelling Bee. It was me against Raj Patel. And I misspelled, in front of the entire school, the word 'failure'.
Dwight: I can't do this.
Michael: That's because you're incapable of doing it because you don't know how. Because you have no skills. Dwight, there's no way I can possibly teach you what you need to know about public speaking by speech time.

Dwight: Oh, okay.

Michael: But I can teach you enough so that you don't embarrass me or the company.

Dwight: Okay, deal! I'll do whatever you say. No questions asked.

Michael: Well, if you have a question, you should ask me.

Dwight: I'll try and think of one. When...

Michael: Don't. Don't try and think of a question to humor me. Just... try not to be such an idiot.

Dwight: Is that an insult or is that part of the public speaking advice?

Michael: Insult.
Pam: Mom, I'm sorry. I know you and Dad are chipping in for the wedding but I do not want orange invitations. Yes! Well, if you really want my...
Jim: Hi, yeah, can I talk to one of your travel agents?
Jim: I'm going to take a trip. I'm going to get out of town for a while... and go someplace... not here.
Jim: Where do I want to go? Um... that is an excellent question. And one I should have probably thought about before I called you. Um...
Oscar: I get here early every morning so I can set the thermostat. I like it a little cooler, around 66 degrees. I'm more productive. Maybe some people don't like it as cold as I do, but I don't care.
Michael: [stand up comic voice] But seriously, what's the difference between a salesman and a saleswoman?
Dwight: Saleswoman has a vagina.

Michael: It's a joke, Dwight. It's not a Sex Ed class.

Dwight: But I'm right?

Michael: Yeah, you're right about the difference between a man and a woman, but not about the punch line to the joke, right? [stand up comic voice] The difference between a salesman and a saleswoman... is boobs!

Dwight: Hey. Do you remember the speeches that you gave?

Michael: I do. Both of them.

Dwight: Could I have a copy of one of them?

Michael: No, no! They would remember them. Look, it doesn't matter what you say. It just matters that you're saying something that people care about. Yeah? All right? Here we go. Watch this.
Michael: Attention everybody! Attention please! I have some very great news from Corporate. We had a wonderful quarter and as a result all of you are getting bonuses for 1000 dollars!
Dwight: Yeah!

Michael: [generalized clapping and cheering] Congratulations.

Phyllis: Unbelievable.
Michael: You see that? You see how they responded to me? In that moment, I had them.
Dwight: That is so great about the bonus!

Michael: No, no! It's not true. I was just talking so just go out there and say anything. They'll eat it up. They're a great audience.
Stanley: Go ahead. Get the wallpaper. Wallpaper the ceiling if you want. Call Terri and tell her she...
Phyllis: It's unbelievable!

Dwight: Excuse me! May I have your attention please? There has been an accident on 84 West. Cars have skidded off the road into the safety railing. Several cars have flipped. There is broken glass everywhere. Several people are injured.

Pam: Do we know anyone who was in the accident?

Dwight: Brad Pitt. Also there will be no bonuses.

Stanley: Why would this affect our bonuses?

Dwight: They are unrelated.

Kelly: Is Brad okay?

Dwight: He will never act again. Also, this branch is closing.

Oscar: What the hell is going on here?

Angela: Are we out of jobs?

Dwight: Yes.

Kelly: This is karma because of what he did to Jennifer Aniston.

Michael: He's kidding. Dwight was kidding and I don't know why because it wasn't funny... and it was just horrible.

Stanley: Michael?

Michael: Yeah.

Stanley: You said we were getting bonuses.

Michael: All right. Everybody in the conference room now. Let's go. Let's do it.

Stanley: Cancel wallpaper.
Michael: As your leader and your friend, I sort of demand that you can all speak in public as I can... and did... twice. [speaking to camera] You saw the plaque, right? [to office] All right. We're all going to go around the room and we're going to make toasts. And that way, we will overcome our fear of public speaking.
Pam: You mean Toastmasters?

Michael: Pam! I'm public speaking. Stop public interrupting me. Actually, this would be good practice for your wedding toast.

Pam: Yeah, the bride doesn't really do... Have you ever been to a wedding?

Jim: Can I go?

Michael: Yes. Good. Jim taking the initiative.

Jim: So. Uh... I am going on a trip. But not really sure where I'm going yet. It's kind of open-ended. So I was hoping maybe you guys would have some suggestions?

Kevin: You should go to Hedonism.

Jim: What is that?

Kevin: It's like Club Med, but everything is naked.

Jim: I was thinking more like Europe. Or something like that. But, good second choice.

Toby: Been to Amsterdam.

Michael: Oh ho hokay. You know what? That's not a toast. You're not standing up.

Toby: [mimes lifting a glass] To Amsterdam.

Jim: When did you go there?

Toby: Umm... After my divorce. Yeah.

Jim: Really for like how long?

Toby: Uh, about a week. Er... .um... .maybe a month. I uh can't...

Creed: Jimmy, listen to me. You do not want to go to Amsterdam. Trust me.

Jim: Where do I want to go?

Creed: I'd send you to Hong Kong.
Creed: Like to say 'Hi' to my friends in China. [speaks in Chinese]
Michael: Okay, Dwight. Show us what you have learned today.
Dwight: Good morning, Vietnam! [general groaning] Okay. You know what? This isn't working. Because um I'm not nervous in front of them. They're my subordinates.

Jim: No. We're not.

Dwight: Uh, yes you are. I'm Assistant Regional Manager.

Jim: Which means absolutely nothing.

Dwight: Michael, can you explain?

Michael: Well, it's mostly made up. So...
Michael: Dwight is not going to do a job. It's sad. And they're expecting excellence because I did do such a good job. Two years in a row. I k*ll. It was amazing.
Michael: Confidence, Dwight.
Jim: Dwight. If you could travel anywhere in the world where would you go?
Dwight: I can travel anywhere except Cuba. And I will travel to New Zealand. And walk the 'Lord of the Rings' trail to Mordor. And then I will hike Mount Doom. So... no... just leave me alone.

Jim: Okay. Just trying to get some advice on my trip.

Dwight: Oh please! You're not taking any trip.

Jim: You know I majored in Public Speaking in College.

Dwight: You did?

Jim: Mmmhmm. And the first thing they teach you is that you've got to be true to your self. And you are all about authority.

Dwight: Yes. I am.

Jim: The great speakers throughout history were not joke tellers. They were people of passion. So if you want to do well today, you got to do what they did.

Dwight: Which is?

Jim: You've got to wave your arms and you've got to pound your fists. Many times. It's supposed to emphasize your point.
Jim: Ok, I didn't actually major in Public Speaking. But, I did download speeches from some of history's famous dictators. Like this one [holds up paper]. Originally given by Benito Mussolini.
Jim: Ok, look. I know you are giving this speech on your own but I wrote up a few talking points for you to take a look at. I hope you don't mind.
Dwight: I'll glance at it.
Michael: It's time, Dwight. The grim reaper is here.
Angela: The very best of luck to you, Dwight.

Dwight: Thank you, Angela.
Kelly: Why'd you pick the V.A. for the reception?
Pam: Roy has a connection. It's nicer than you think.

Ryan: You're inviting Jim?

Pam: Of course. He's one of my closest friends.
Michael: All right. You ready? Here we go! Wow. It's a little bit bigger than I remember. Come on. We're down here. Right.
Overhead: [song] You all ready for this?
Angela: [coughs] [sniffles] I am just feeling under the weather. And... I think that I will go home and rest.
Kevin: I've never, ever seen you take a sick day.

Angela: Well, I've seen you take enough for the both of us.
Speaker: Next, I'd like to introduce the Dunder Mifflin Salesman of the Year, Dwight Schrute!
Crowd: [polite clapping]

Michael: Dwight, they called your name.

Speaker: Dwight, how we doing?

Dwight: No, I can't... I ca...

Michael: All right. You know what? Okay. No. No problem. You are lucky you have me here. I'm going to cover for you. [shouts] All right!

Crowd: [claps]

Michael: Gooood morning, Vietnaaaam! I am not Dwight Schrute. Not at all. I am Michael Scott, his mentor and boss. And until Dwight comes up, if he ever does, I wanted to say a few words about excellence. What makes a work environment excellent? Well, there are many things, I believe, that do such a thing of that nature. And one would be humor. What is the difference between a salesman and a saleswoman?
Kevin: I always set it at 69. [snickers]
Pam: Maybe we'll use a DJ. That's the one thing Roy's in charge of for this wedding but all he's managed to do is set a date.
Kelly: But he did a great job. June 10th is perfect. I want a June wedding. I've always wanted one. Ryan, do you know when you would want to get married?

Ryan: Actually, I don't see myself ever getting married.

Kelly: Oh.

Pam: Ryan, you should be more sensitive. It's obvious she likes you and comments like that, they just...

Ryan: I know what I said.
Michael: I'm very sorry. I did not know you were wearing a hearing aid and I just thought you were speaking abnormally. ...And now the black guy from the 'Police Academy' movies. A robot. [makes robot sounds] Michael Winslow, anyone?
Michael: Car starting. [makes car sounds] All right, Dwight Schrute everyone.
Crowd: [clapping]

Michael: Good luck. That is a tough crowd.

Dwight: [bangs fists] Blood alone moves the wheels of history! Have you ever asked yourselves in an hour of meditation, which everyone finds during the day. [waves arm] how long we have been striving for greatness? [bangs fist] Not only the years we've been at w*r, the w*r of work, but from the moment as a child when we realized that the world could be conquered. It has been a lifetime's struggle [waves arms]. A never-ending f*ght. I say to you [hits podium] and you'll understand that it is a privilege to f*ght!

Crowd: [clapping]

Dwight: WE ARE WARRIORS!

Crowd: [clapping and cheering]

Dwight: Salesman of Northeastern Pennsylvania, I ask you once more rise and be worthy of this historical hour!

Crowd: [clapping and cheering]

Dwight: [laughs maniacally] Yeah. Yes!
Oscar: I've got a time share in Key West that might be available.
Jim: Maybe. Thanks.

Ryan: You really think you're going to go?

Jim: Yeah. I'm definitely going.

Ryan: Nice. Send me a postcard.
Ryan: Jim has worked at the same place for five years. Jim eats the same ham and cheese sandwich everyday for lunch. I don't know. If I were a betting man, I'd say he will have a fun weekend in Philadelphia.
Dwight: No revolution is worth anything unless it can defend itself. [bangs fists]
Crowd: [claps]

Dwight: Some people will tell you salesman is a bad word. They'll conjure up images of used car dealers and door to door charlatans. This is our duty - to change their perception. I say salesmen... and women of the world unite! We must never acquiesce for it is together, TOGETHER, THAT WE PREVAIL! We must never cede control of the motherland! For it is...

Crowd: [shouts] Together that we prevail! [cheering and clapping]
Pam: Australia? I have always wanted to go there?
Jim: I'm going. I'm a little nervous to run into Dwight on his connecting flight to Mordor. But, other than that... um, yeah, I bought the ticket, non-refundable.

Pam: That's awesome. Where are you staying?

Jim: I don't know. I feel like I have plenty of time to figure out the details but...

Pam: When are you leaving?

Jim: I'm... leaving on June 8th.

Pam: Oh.

Jim: Yeah. And I'm really sorry about that, I just...

Pam: Oh yeah. That's too bad.

Jim: Yeah. Do you want me to take these on my way out?

Pam: It's ok. I got it.

Jim: Alright.
Dwight: Ok, thanks. [to Michael] There you are. What happened?
Michael: I got thirsty. How'd it go?

Dwight: It was amazing. I wish you would have been there.

Michael: You would not believe what happened here.

Dwight: What? Something happened?

Michael: Oh! This woman came in, sat down, ordered a drink. The bartender asked for her ID which I thought was odd because I pegged her at like 35.

Dwight: Weird.

Michael: Yeah, it was weird. So, she was like 'I don't have my ID, please give me one.' And he was like 'I can't do that. I can't serve you.'

Dwight: Con artist.

Michael: She might have been. So she says 'Fine. I will go to my room. I will get my purse. I will come back. I'll show you my ID.' She hasn't come back yet. She's probably in her room drinking from the mini-bar! Right?
Michael: Dwight gave a great speech. That's the word on the street anyway. And I entertained Dwight to no end with my bar stories. So, I captivated the guy who captivated a thousand guys. Can you believe that? A thousand guys?
Pam: I'm looking forward to 'Take Your Daughter to Work' day. I am not great with kids, but I wanna get better. Because I'm getting married. So, I put out a bunch of extra candy out on my desk so the kids will come talk to me. ...Like the witch in Hanzel and Gretel.
Jim: Bribery. Nice.
Pam: Oh, I have more. [Holds up bags of candy]

Michael: Pam. Ms. Beasley if yer nastay! Janet Jackson. Hey! You having a wardrobe malfunction there? Or w---

Pam: Oh, Michael. You can't be nasty today. [whispering] 'Cause of the... [points to 'Welcome Daughters!' sign]

Michael: ... Oh, God is that today?

Pam: I reminded you last night.
Michael: Listen, I like kids. But this is not a kid's environment. This is like HBO, no limits. Who knows what I'm going to say? Crazy stuff. And it is R rated, it is not rated G. I am like Eddie Murphy in "Raw," and they are trying to make me into Eddie Murphy in "Daddy Daycare." both great movies, but, still.
Michael: Well, I'll be in my office.
Pam: Don't you think you should say something?

Michael: They're cool.

Pam: Michael, I think that as the boss you should really---

Michael: Fine, fine, fine, fine, fine, fine, fine. Hi, children. I'm Michael Scott, and... I... am in charge of this place... ahh, what'll make you... understand... I am... like Superman, and the people who work here are like citizens of Gotham City.

Jim and Dwight: [in unison] That's Batman.

Michael: Okay, I'm Aquaman. Where does he live, guys?

Jim: The ocean.

Michael: [under his breath] I work with a bunch of nerds.
Dwight: [looks at Sasha] Mmm... hello tiny one.
Toby: [to Sasha] Come on.

Dwight: You are the future!
Kevin: This... is my file cabinet. Uhm... oh. This... is the partition... between my desk... and Angela's.
Kevin: Abby's my fiancee Stacy's daughter, I think she'll have a good time. I just hope she doesn't look on my computer. ...Actually, I'd better go check.
Stanley: Michael, you remember my daughter, Melissa.
Michael: Oh, yes, hello, how are you? Good to see you. Wow, you've really grown up. You know what? Don't mind me saying so, she is turning into a stone cold fox. Better keep the... frat boys away from her.

Melissa: I'm in eighth grade.

Michael: Oh.

Stanley: She's in middle school.

Michael: Yeah, middle school's amazing. It is extraordinary. An extraordinary time.
Michael: It's not that children make me uncomfortable, it's just that, why be a dad when you can be a fun uncle? I've never heard of anyone rebelling against their fun uncle.
Michael: [while Jake is throwing things at Michael] They want how many spiral pads?
Meredith: Um, fif--well, fifty... I... over ordered because they had a back order.

Michael: Okay.
Meredith: I got permission to bring Jake into work, which is great because he got suspended this week and now I don't have to pay for a sitter.
Angela: Can you put that down there?
Kelly: Yep. [spreads tablecloth]

Toby: [to Sasha] Okay, tell them what you wanted to say.

Sasha: Do you need any help?

Angela: No. Thanks. We'd... have to explain everything, it's probably just easier if we do it ourselves.

Toby: Alright, I wasn't expecting that. Let's uh... let's go draw.

Kelly: Oh my God, she is so cute, I want to die. Don't you just love kids, Angela?

Angela: I guess I wouldn't mind a pair of small, well-behaved boys.

Kelly: God I cannot wait to get pregnant and have babies!
Ryan: Kelly and I both agreed that we would just have fun, and, I'm learning that fun for Kelly is... getting married and having babies. Immediately. With me.
Michael: [on phone] Just compare last year's order to this year's. Uh-huh. Yeah, I'm looking at it right now. [Sasha walks in the door] ... Yes. We--yeah, they're very--they're different. [Sasha walks out] Yeah, we can stick with last year's, you're just going to have to supplement it, somehow.
Pam: Hey, Abby! Do you want to help me shred some old documents? It's actually pretty cool.
Abby: No thanks.
Pam: I only have one goal today. To make one kid like me. Just one.
Jim: What are you reading?
Abby: From the Mixed-Up Files of Mrs. Basil E. Frankweiler.

Jim: Aww, best book?

Abby: Yeah, but I've read it before.

Jim: Pfft. So have I. Hey, question. If you had to spend a night in the Met or the Aquarium, which would it be?

Abby: Definitely the Aquarium.

Jim: Definitely. Yes. Glad you said that. ...You don't want to help me with some of my sales, do you? 'Cause, I'm kind of swamped.

Abby: Sure.

Jim: Really?

Abby: Mmhmm!

Jim: Yesss. And you're Abby, right?

Abby: Yeah.

Jim: I'm Jim. [Jim hi-fives Abby] Annnnd... let's sell some paper.

Abby: Alright.

Jim: Let's start with... your mom.
Michael: [on phone] Yes. Well... we can... [Sasha walks in, begins playing with Michael's toy train] uhm... hey, uh, you know what? Can I call you back? I'll call you right back. Yes, I promise. ...Hello, can I help you? ... You can pick that up, if you want. That's--- that's alright. [Sasha moves the train to Michael's desk] Want to bring it over... here, make some room. My name's Michael. What's your name?
Sasha: Sasha.

Michael: Nice to meet you.

Sasha: Ooh! [picks up train whistle]

Michael: Oh, you know what that is! That is a train whistle, like I'm the conductor. [blows into whistle] But I'm sort of the conductor of the office here, right? [blows into whistle] You want to try?

Sasha: Sure. [Sasha blows into whistle continously]

Michael: All aboard for sales! Next stop, Cu...camonga! [Sasha and Michael laugh]
Jim: [shaking hands with Abby] Ow, ow, ow, ow, you broke my hand.
Dwight: There is no way that hurt.

Jim: Really? 'Cause she's pretty strong, Dwight.

Dwight: Little girl. Come over here. Shake my hand. Come on, I don't have all day. [Abby shakes his hand] I don't feel anything. Nothing. [to Jim] You're so weak. [Jake walks over and messes with Dwight's bobbleheads] Uh, excuse me, these are expensive collector's items, okay?

Jake: Do you have any computer games?

Dwight: No, I don't have computer games on my work computer. That would be innappropriate.

Jake: Yeah, Meredith doesn't have any either. It's so lame here.

Dwight: You call your mom Meredith? That's very disrespectful.

Jake: Whatever, okay?

Dwight: You can refer to me as Mister Schrute.

Jake: That's your name? Mister Poop?

Dwight: Schrute. Mister Schrute.

Jake: Sure, Mister Poop. [Jake walks away]

Dwight: [quietly] ... Schrute. [Jim and Abby snicker, Angela glares at Dwight]
Sasha: [to Phyllis] Are you Mother Goose?
Melissa: I drink like, a hundred Ice Macchiatos a day, and practically nothing else.
Ryan: Wow.

Melissa: There's a really cool coffee place, Jitters, at the Steamtown Mall. Ever been there?

Ryan: No.

Melissa: You've never been to Jitters? Ryan, you are so dorky. Gimme your number, so I can text you.

Ryan: Uhm...

Melissa: Come on! [Kelly glares through the door] You have an email address?
Kelly: ...that I thought you should know ...
Stanley: Mmhmm. What?

Kelly: I think something a little fishy is going on. [points to Ryan and Melissa]

Stanley: A little fishy?

Kelly: Yeah. I mean, I've been noticing them all day, I was thinking that maybe ... [Stanley gets up]
Stanley: That little girl is a child! I don't want to see you sniffing around her anymore this afternoon, do you understand?!
Ryan: Yes, I--

Stanley: Boy have you lost your mind? 'Cause I'll help you find it! Whatcha lookin' for, ain't nobody gonna help you out there! Jesus could come through that door and he's not gonna help you if you don't stop sniffing after my child!

Ryan: Okay.
Ryan: Stanley yelled at me today. That was one of the most frightening experiences of my life.
Dwight: [plays the recorder] That was Greensleeves. A traditional English Ballad about the beheaded Anne Boleyn. And now, a very special treat... a book my Grandmutter used to read me when I was a kid. This is a very special story, it's called Struwwelpeter, by Heinrich Hoffman from 1864. [reading from book] The great tall tailor always comes to little girls that suck their thumbs--- are you listening, Sasha? Right? And 'ere they dream when he's about, he takes his great sharp scissors out, and then cuts their thumbs clean off!
Michael: Dwight! Dwight!

Dwight: There's a photo...

Michael: What the hell are you reading to them?

Dwight: These are cautionary tales for kids, my Grandmata used to read these---

Michael: Yeah, you know what? No, no no no no. They, no. The kids don't want to hear some wierdo book that your n*zi w*r criminal grandmother gave you.

Sasha: What's a n*zi?

Michael: What's a n*zi?

Dwight: [standing up] n*zi was a fascist movement...

Michael: Don't!

Dwight: ...from the 1930's...

Michael: Don't! Don't! Don't talk about n*zi in front of--- you know what? They're going to have nightmares, so why don't you just shut it?

Dwight: I was gonna teach the children how to make corn-husk dolls.

Michael: [sighing] Why don't you just leave? Okay?

Dwight: ...Okay.

Jake: Bye, Mister Poop.

Michael: Alright. There goes Mister Poop. Now, who likes Dane Cook?

The Kids: [raising hands] I do, I do!
Michael: Children cannot lie. They are innocent, and they speak the truth, and out of the mouths of babes, Michael Scott is freaking cool. [cracks up]
Angela: You know, I never misbehaved in front of my father because he was a very strict disciplinarian. I can only hope my mate has some of those same qualities [makes eye contact with Dwight].
Michael: This is where the magic happens! Right over here, let me show you this. See all these? [pets shelf of paper] You know what that is? That's paper. This is where paper comes from. Any questions?
Melissa: So... you cut the paper and dye it and stuff.

Michael: No, we don't actually cut the paper. That's a good question. The paper is sent to us cut, and dyed, from a paper manufacturer, and then we sell it to a business for more than we paid for it.

Abby: That's not fair. [the rest of the kids agree]

Michael: Yes it is, well, w-w--you need someone in the middle to facilitate...

Jake: You're just a middleman.

Michael: I'm not just a middle... man...

Melissa: Wait, why doesn't the saw mill just sell paper directly to people?

Michael: You are describing Office Depot, and they're kind of running us out of business.

Dwight: We have better service than they do!

Michael: ...There's Creed! Let's take a look at what he's doing, everybody! This is Creed, and he is in charge of... something. Right?

Creed: That is correct.

Michael: Say hi to the kids.

Creed: Hi kids.

Michael: Yaaaaay.

Creed: Have you ever seen a foot with four toes? [begins untying shoe]

Kids: Ewwww!

Michael: What are you doing? N--stop it! Stop it! Just--no, no, no, no! No! Would you cut it out?! What is your problem?

Creed: Th-the hair covers it, mostly.

Michael: No no no, we're not gonna see--- we're not gonna see the four toed... Creed, okay?
Michael: You know, there's something interesting about me you might want to know. I ... used to be ... the star of a kids show.
Kids: No way.

Michael: It's true. I did.

Melissa: You serious?

Jake: Really?

Michael: I am totally serious. There was a show called 'Fundle Bundle' and I was the star.

Abby: That doesn't sound like a show.

Melissa: What?!

Michael: It's true! I can prove it! I can prove it, watch this. [gets up and runs out] Ryan, can you come here a second? [clears throat] I would like you to go to my mother's house in Dickson city, and if she is in the pool, the back kitchen window should be unlocked, I want you to boost yourself up, I want you to go down to the basement. In the basement is a tape labeled 'Fundle Bundle'. I want you to grab it, I want you to get my guitar.

Ryan: Right. Okay.

Michael: I want you to get the tambourine. Do you know how to play the tambourine?

Ryan: Um, I'm already getting the pizzas from Bernetti's, so...

Melissa: You know, I can go with him.

Michael: Oka--

Ryan: No! I will... go.

Michael: Okay! Thank you Ryan. Good attitude, hottest in the office.
Michael: [to Abby] Alright, nowwww... what kind... of pizza do you like?
Michael: I don't get why parents are always complaining about how tough it is to raise kids. You joke around with them, you give them pizza, you give them candy, you let them live their lives... They're adults, for God's sake.
Michael: I am going to give you a little blast from the past of Michael Gary Scott when he was a child star, and a show that you might remember called 'Fundle Bundle.' Okay? Without further ado, Ryan?
Miss Trudy: [from TV] ...Bundle, are you ready to come on in? [TV children cheer] Let's have some fun!

Michael: That... is Miss Trudy. Can't tell from the costume, but she had an amazing body. Okay, you can... fast forward. And... I want you... to...

Dwight: Is that a real fun sh**ting windmill?

Michael: Stop! Stop! Stop! [Ryan resumes the tape] Yes! That is, uh, Edward R. Meow.

Jim: That's pretty funny.

Michael: Yeah.

Edward R. Meow: ...Recess! Hey, what's your name?

Chet: My name's Chet.

Edward R. Meow: Well hi Chet.

Oscar: Is that Chet Montgomery?

Michael: Uhh, I don't know.

Pam: That is!

Darryl: Checkin' in with Chet. Doppler's up.

Edward R. Meow: What do you want to be when you grow up?

Chet: I want to be on TV!

Dwight: [employees chuckle] And he is on TV now!

Michael: Can everyone please shut up, please! So you don't miss it.

Edward R. Meow: Okay, next? So, whats your name?

Michael: Oh! That's me. Shh. Shh.

Edward R. Meow: Well what's your favorite subject at school?

Young Michael: Recess.

Edward R. Meow: Ree-cesss. So tell me, what do you want to be when you grow up?

Young Michael: I want to be married and have a hundred kids so I can have a hundred friends, and no one can say no to being my friend.

Edward R. Meow: [jaw drops, awkward pause] Uh, ah... oh, okay! Well uh, nice talking with you, Michael. Uh, back to you Miss Trudy!

Miss Trudy: Hi everyone, it's one of my favorite times of the day.

Michael: Coulda sworn there was...

Melissa: Did you get married?

Michael: ...uh, no.

Abby: Why not?

Michael: Uh, just never happened.

Sasha: So, do you have any kids?

Michael: Uh, nope.

Jake: Do you have a girlfriend?

Michael: I do okay.

Melissa: Was Chet Montgomery cool back then?

Michael: Yes.

Jake: Even I have a girlfriend.

Michael: Okay! Alright, okay.

Sasha: So you didn't get to be what you wanted to be.

Michael: ...I guess not... you know, I have a load of work to do so I am going to grab a slice of this delish pizza... and I'm going to go do my work. Bye.
Pam: He's not coming out. He won't pick up the phone.
Jim: Can't believe his mom dressed him like that, that's the real tragedy.

Roy: [wrestling with Jake] Pam! Pam! I love this guy! [laughs] Come on!
Pam: So, Melissa... I met your mom a couple times. She's so nice.
Melissa: Who? Terry?

Pam: Mmhmm.

Melissa: That woman is not my mother. That is my step-mother.
Jake: Mister Poop, I have to tell you something.
Dwight: Uh, okay. But first, that's not my name.

Jake: You're ugly.

Dwight: Well at least I'm not a horrible little latchkey kid who got suspended from school. So...

Jake: Meredith!
Michael: [Toby knocks on door] Yeah?
Toby: I think these belong to you. [puts toys down on desk]

Michael: Oh, that's okay, she can keep those.

Toby: Believe me, she has enough toys... she doesn't need your watch.

Michael: Thank you.

Toby: Is everything okay?

Michael: You have to ask me that because you work for human resources.

Toby: Uh... it's true...
Michael: Well, sure, playing the field is great, don't get me wrong, but there's more to life than notches just on my bedpost.
Toby: Mmhmm.

Michael: Tell me something honestly, do you... think... that it is too late for me to have kids?

Toby: Well, you need a wife first, or at least a girlfriend.

Michael: What about...

Toby: Not Jan.

Michael: ...Jan. Kay.

Toby: If you really want to have kids, I--- I guess you could somehow... foster parent, or something.

Michael: ...Or biologically.

Toby: Somehow.

Michael: Thanks, that's, no, that... that really means a lot to me. Hey, does Sasha have a godfather, because I...

Toby: Yes.

Michael: Oh... kay.
Jake: Is it okay if I take one?
Pam: Sure.

Jake: Thank you.

Pam: You're welcome.

Jake: Is your job hard?

Pam: It's not too bad. I get to shred things sometimes, do you want to see?

Jake: Yeah!

Pam: Really?

Jake: Yeah.

Pam: Okay. Um... here it is. Don't put your fingers in there. [shreds paper] Cool huh?

Jake: That's so cool, yeah!

Pam: Yeah, I get to do this like, every week.

Jake: That's so awesome!

Pam: I know.
Michael: Yes, it is true. I, Michael Scott, am signing up with an online dating service. Thousands of people have done it, and I am going to do it. I need a username. And... I have a great one [types]. Little kid lover. That way, people will know exactly where my priorities are at.
Kevin: Go ahead.
Abby: Do you want to come over for dinner tonight?

Jim: Ohh, man, I would love to! I can't tonight, but can I come over some other time? [Abby nods]

Kevin: What're you doing? You never have plans.

Jim: Thanks, Kev. Uhm... I'm actually going on a date.

Kevin: Niiice.

Michael: Hey, uh, no, please? You can't leave yet. There's still one more thing we need to do.

Michael: [singing] You... who are on the road... must have a code... that you can live by... [Dwight joins in] and so... become yourself... because the past... is just a goodbye... and teach... your children well...

Jim: Why does he own a guitar if he doesn't know how to play?

Pam: I think he thought his ukulele skills would transfer. [Jim leaves]

Michael and Dwight: [singing] ...did slowly go by... and feed... them on your dreams...

Pam: My theory is that... [Jim signals he's leaving, waves bye to Pam]

Michael and Dwight: [singing] ...The one they picked... the one you'll know by... don't you ever ask them why... if I told you would cry... you never look at them and sigh... and know they love you...

Dwight: And they do, your parents, love you very much.

Michael: One more time. [singing] You...
Dwight: The Schrutes consider children very valuable. In the olden days, the women would bear many children, so we would have enough laborers to work the fields. And if it was an especially cold winter, and there weren't enough grains or vegetables, they would eat the weakest of the brood. [Laughs] They didn't eat the children.
Michael: So, Phil recruited me to sell these cards, and now I am recruiting you.
Oscar: Who is this guy again?

Michael: Don't worry about Phil. He drives a corvette. He is doing just fine. Okay. Calling cards are... the wave of the future. These things sell themselves.

Ryan: Who uses calling cards anymore?

Michael: You know what? That's a nice attitude, Ryan, I'm just helping you invest in your future, my friend.

Oscar: This sounds like a get rich quick scheme.

Michael: Yes! Thank you! You will get rich quick. We all will!

Toby: Didn't you lose a lot of money on that other investment, the one from the email?

Michael: You know what, Toby? When the son of the deposed king of Nigeria emails you directly, asking for help, you help! His father ran the freaking country, okay? ...Alright, so, raise your hand if you wanna get rich. [Jim and Dwight raise their hands] Alright.

Jim: No, um. How is this not a pyramid scheme?

Michael: Alright, let me explain. Again. [draws on board] Phil has recruited me and another guy. Now, we are getting three people each. The more people that get involved, the more who are investing, the more money we're all going to make. It's not a pyramid scheme, it is a... it's not even a scheme per se, it's... [Jim draws a triangle around Michael's diagram] ... I have to go make a call.
Pam: Happy birthday Michael.
Michael: Oh ho ho! What?

Pam: I said happy birthday.

Michael: Thank you! That's really nice.
Michael: Today is my B-day, and people around here just go crazy for it. I don't know why. Oh! Fun fact. I share my birthday with Eva Longoria. So, I have a perfect ice-breaker if I ever meet Terry Hatcher.
Michael: What's up?
Jim: Hey. ...Oh, happy birthday.

Michael: Ah, thank you sir.
Meredith: Did you hear anything yet?
Kevin: No. I'm still waiting.
Michael: [Dwight knocks on door] Yeah.
Dwight: Yes. There he is, the birthday boy!

Michael: Ohh, god.

Dwight: Birthday hug!

Michael: No no no, no, new suit, please.

Dwight: That suit is amazing.

Michael: Thank you very much. It is from Italy. [checks jacket] Actually--- no, Bulgaria.

Dwight: Mmm. Maybe I should get one.

Michael: Good luck. One of a kind.

Dwight: Ebay. Hm. Question! May I be in charge of the party planning festivities?

Michael: Not necessary, the party planning committee is all over it. They've been working twenty-four seven all day yesterday.

Dwight: Excellent. On my part, I did manage to reserve the...

Michael: Don't! Nope! Please, don't want to spoil it for anybody. Spoil the surprise.

Dwight: Let's get the party started. [Begins 'raising the roof']

Michael: Let's get the party started. Not the way I taught you! [Michael joins in]
Phyllis: When should we bring out the cake, one or one thirty?
Pam: One's good.

Angela: One thirty. [Pam yawns] I'm sorry, are we boring you?

Dwight: Party planning committee, listen up. Michael would like trick candles for his birthday cake, so make that a priority.

Phyllis: Where do we get those?

Dwight: Not my problem. Here is a list of things that Michael would like to be surprised by. [hands list to Pam]

Pam: Michael wants a strippergram?

Dwight: Yes, but he doesn't want to know when, or whom.

Angela: No. This is a closed door meeting.
Michael: [answering phone] Yeah?
Pam: Michael, I have Jan on the line.

Michael: Oh, great, put her through.

Jan: Hello, Michael.

Michael: Hey, you.

Jan: I'm... returning your call, you said it was urgent.

Michael: It is urgent, I just wanted to call and wish you a happy birthday.

Jan: Well, today's not my birthday, so...

Michael: Really? 'Cause, I thought we had the same birthday.

Jan: ...Happy birthday, Michael.

Michael: Thanks. [grins]

Jan: Am I on camera?

Michael: Nope. Totally private. You can say whatever is in your heart. [Jan hangs up]

Michael: [to Ryan, sitting across from Michael] ...You can take a five, if you want.
Michael: Somebody brought in donuts for my birthday!
Stanley: Mmhmm, happy birthday.

Michael: Thanks.
Jim: Man, I'm so sorry. When do you find out?
Kevin: They said this afternoon. They're waiting on a second opinion.

Jim: Oh, okay.

Kelly: Second opinion on what?

Kevin: Um, I might have skin cancer.

Kelly: Oh, no! I was watching Grey's Anatomy, and there was a lifeguard on it, and he had skin cancer too.

Jim: Kelly, you know what...
Kelly: I never really thought about death until Princess Diana died. That was the saddest funeral ever. That and my sister's.
Toby: Who brought in donuts?
Michael: Somebody got donuts for my birthday!

Toby: Happy birthday!

Michael: You didn't know it was my birthday.

Toby: I... guess I forgot.

Michael: Well, I guess I forgot to give you a donut [closes box].

Toby: Are you serious?

Michael: Mmm.
Oscar: Skin cancer is treatable.
Kevin: Right.

Oscar: It's going to be okay.

Angela: You don't know it's going to be okay. Don't give him false hope. ...It's probably nothing, though.
Delivery Woman: Hi, delivery for Michael Scott.
Michael: Here we go. Ohhhkay, this is great! [giggles] Thank you my friends, she is perfect! Ahhh, Dwight, may I have your chair please? And, um, some singles, if you will! Allllright. Nnnnn-dink! [puts single into delivery woman's pocket, giggles] Okay, um, alright. This has arms. Is that gonna be a... is that alright?

Delivery Woman: Uh... s-sure.

Michael: [laughing] Okay. I'm so nervous.

Pam: I can sign for it.

Delivery Woman: Oh. Thanks.
Michael: When I was seven, my mother hired a pony and a cart to come to my house for all the kids... and... I got a really bad rash from the pony, and all the kids got to ride the pony and I had to go inside, and my mother was rubbing cream on me... for probably three hours, and I never came outside. And by the time I got out the pony was already in the truck. And around the corner. So that was my worst birthday.
Michael: [eating donuts while Dwight plays the recorder] Stop it. Stop! What is that?
Dwight: It's 'For the Longest Time,' by William Joel. It's you favorite song.

Michael: Yeah, well, it's on the radio. My birthday blows. Nobody even signed my birthday poster. Probably my mother is the only one that cares enough to send me anything.

Dwight: I probably care more than she does.

Michael: You're making it worse. I bet Luke Perry's friends don't treat him like this [points to James Dean poster].
Pam: When does he hear?
Jim: Sometime today.

Pam: Ohh... poor Kevin.
Pam: If I knew I had a week to live, I would... probably go to Europe. And South America. And the Grand Canyon. And... I would want to see the Pacific Ocean. ...It would be a pretty busy week.
Dwight: Uh, that's a list price of four dollars and fifty cents. Unfortunately, this item is on [watch beeps] back... order... [hangs up] Michael! Michael! Michael Michael Michael! Come here, come here, come here! Come here!
Michael: What?

Dwight: Listen up everyone! It is 11:23 exactly, the exact moment when you emerged from your mother's vaginal canal, so... huh?! Right, have a seat. Please.

Michael: [grinning] Ohhh, God.

Dwight: There is a tradition that the Hebrews have of hoisting the birthday boy up on a chair.

Michael: Ohhh, no.

Dwight: So come help me celebrate Michael's birth moment. Kevin!

Oscar: ...I'll do it.

Michael: Ohh, no, no, no! I can't... Ryan, come on. Let's do this.

Dwight: Creed! Come on. Stanley!

Pam: ...I feel like we should go get Kevin something. Do you think we can sneak out of here?

Jim: Maybe, but... we're gonna need somebody to create a diversion, and...

Dwight: On three, we're going to hoist away! Ready?

Michael: Okay.

Dwight: Happy birth moment, Michael.

Michael: Thank you.

Dwight: One. Two. Three! [Michael is raised until his head hits the ceiling]

Michael: Whoa whoa! Alright. Alright. Watch it... please.

Dwight: Oscar...

Oscar: It wasn't me.
Dwight: Okay, that is not an eight foot sub.
Delivery Boy: Uh, we don't make an eight foot sub. This is eight one foot subs.

Dwight: F. Alright, what's the damage?

Delivery Boy: Uh, thirty-nine sixty.

Dwight: [pulls out wallet] Thirty nine... sixty.
Dwight: Why tip someone for a job I'm capable of doing myself? I can deliver food. I can drive a taxi. I can, and do, cut my own hair. I did, however, tip my urologist, because... I am unable to pulverize my own kidney stones.
Dwight: Here they come.
Michael: Get in here... everybody.

Dwight: Come and get it!

Michael: Birthday party subs! My gift to you.

Oscar: What is this?

Dwight: Uh, bologna, tomato and ketchup.

Michael: The best.

Stanley: These are all the same?

Michael: Yes.

Angela: Bologna? I don't eat bologna.

Michael: Well, then just have the tomato and ketchup. Still good.

Angela: No.

Michael: Just the bread, it's fresh baked.

Angela: No.

Michael: Mm-kay. Get whatever you want. [under breath] And choke on it.
Michael: When I was sixteen, I was supposed to go out on a date with a girl named Julie. But there was another Michael in the class that she apparently thought the date was with, so she went out with him, on my birthday. And, she got him a cake, at the restaurant. And it wasn't even his birthday, but I heard about it the next day in school. So... That was the worst birthday I think I ever had.
Jim: So. We got Kev some stuff. Um... a party pack of M&M's, his favorite candy. A DVD of American Pie 2, which is his favorite movie, and, he lent it to Creed, so, I can guarantee you he won't get that back.
Pam: Sixty-nine cup of noodles.

Jim: Which we realize sounds crass, but, it... is his favorite number.

Pam: And his favorite lunch.
Dwight: Hey temp, you know uh, we still got five feet of sandwich left [pulls ice cream cake out of freezer].
Ryan: [making peanut butter and jelly sandwich] Someone ate three feet of that thing?

Dwight: Hell, yeah. Save room for ice cream cake.

Angela: [grabs cake] Oh. Thank you.

Dwight: Oh. I got it.

Angela: What are--- it's... the party planning committee.

Dwight: [whispering] This is the most important day of the year. I can't risk anything.

Angela: Fine.

Dwight: What about that meeting... later... to discuss finances?

Angela: Yes... [whispering] but don't expect any cookie.

Dwight: [whispering] But what if i'm hungry?

Angela: [whispering] No cookie.
Jim: [puts fabric softener into cart] ...What?
Pam: You use fabric softener?

Jim: Yeah, you don't?

Pam: No, I do.

Jim: ...Okay.
Office Staff: [singing] Happy birthday dear Michael, [Michael joins in, Kevin's phone rings] Happy birthday... [everyone but Dwight stops] ...tooo youuuu.
Kevin: Hello? Hey.

Michael: Kevin? Respect the birthday please.

Kevin: No, um, no not yet. I will. Bye. [hangs up] It was just Stacy.

Michael: Are you done? ...Good. Okay.

Dwight: Here we go. Make a wish.

Michael: Uhhh... blow out the candle. Okay. Mmmm... [blows out the candles]

Dwight: Yaoo yay! [claps]

Michael: ... I asked for trick candles.

Dwight: Pam was supposed to get 'em.

Michael: Okay. Well, when she comes back we'll do it again. [notices Meredith hugging Kevin] Hello, what about the birthday boy? Haven't had a hug all day.

Angela: No one cares about your birthday. Kevin's waiting to hear if he has skin cancer.

Michael: ... Aww, that... sucks, great. ... Wow, that's good timing. That's... that's, sorry, that's terrible. Terrible news. That's terrible... terrible news for both of us [takes cake into office and slams the door].
Pam: [checking watch] We should probably head back.
Jim: Yeah. Okay. Oh. I dare you to make an announcement.

Pam: You dare me? How old are you?

Jim: Just... quit stalling.

Pam: [over loudspeaker, imitating Darth Vader] Luke, this is your father. Come set the table for dinner.

Jim: Such a dork.

Pam: [loudspeaker] Jim Halpert? Price check on fabric softener, the kind that gives you...

Store Employee: Ma'am? Please don't touch that. That is not a toy.

Pam: Oh I'm sorry. I'm sorry.

Jim: How old are you?

Pam: I hate you.
Toby: [to Kevin] Honestly, is there any way you can get on your fiancee's plan? Our health plan is s... just... it's terrible.
Michael: There you are. Good news. Did some research. It turns out that 98% of people with skin cancer fully recover.

Kevin: Still scary.

Michael: Yeah, but it's not brain cancer. And it shouldn't stop us from having fun. You know what they say the best medicine is.

Kevin: Well the doctor said a combination of interferon and dicarbazine.

Michael: And laughter... also.

Toby: I don't really think people are in the laughing mood.

Michael: Why are you here? I didn't even invite you to my birthday party.

Toby: I work here.

Michael: [mocking voice] Nyeh, I work here. [to Kevin] Alright, well, you know what, since Toby doesn't speak for everybody and I am your boss, I... think you should just go home. Take the rest of the afternoon off. Take a sick day.

Kevin: If I go home now, I'll just drive myself crazy.

Michael: Well, you're pretty much driving everyone else here crazy... crazy with worry.
Dwight: Where have you been? And don't say the bathroom, 'cause I kicked in all the stalls.
Jim: Well that's an invasion of privacy, so, I'm going to tell Michael.

Dwight: Please, don't.

Jim: You... owe me.

Michael: Excuse me, everyone. Attention please. Kevin, we're going to take you to a very special place, a place that will make you happy, and a place that is far, far away from the evil sun.

Stanley: Is this trip related in any way to your... birthday?

Michael: How dare you sir. You are gross.
Michael: [sees 'Happy Birthday Michael Scott!' poster at skating rink] That should not be there.
Dwight: I'll get someone to take it down.

Michael: No, it's alright. It's already up. Just leave it. Where's Kevin? Come on! Let's get our skate on!
Kelly: Don't be scared! You're good! You're good!
Ryan: Whoa whoa whoa whoa whoa whoa whoa.
Jim: Think you can let go?
Pam: No. [laughs]

Jim: Whoa whoa whoa whoa whoa. [Michael skates by]

Dwight: YEAH!

Pam: Who is that?

Jim: Is that Michael?
Michael: Yeah, I've been pretty much skating my whole life. I thought about playing in the NHL, but, you're on the road so much. You got no time to spend with your wife and kids. And I really want a wife and kids.
Pam: I got it.
Michael: Hey Pam, all this stuff with Kevin... um, it's pretty scary. And I'm thinking that uh, next time you're in the shower, you should check yourself out. You know, give yourself an exam. Those things are like ticking time bags. Alright? Think about it.

Jim: ...It's something to think about.
Kevin: I can't relax about it, you know?
Michael: Kevin. You heard anything yet?

Kevin: No, not yet.

Michael: Okay. Well. Live strong.

Kevin: Okay, Michael.

Michael: Alright.

Carol: Michael?

Michael: Yeah. Carol? She sold me my condo! Hey! What, is this place on the market? Or...

Carol: Uhh, no, I... don't just sell real estate. Uh, my daughter has a skating lesson.

Michael: Oh, these... all your kids?

Carol: No just the front two.

Michael: Oh, hey guys. Whats up? You wanna go for a ride? Is that okay?

Carol: Sure.

Michael: Cool. Alright. Grab on. Here we go. Ready? Hang on tight. Alright. We are moving. We are reaaallly mooovin' now!
Michael: Push. Good! That's great. You got it. [Kevin's phone rings] Excuse me.
Kevin: Hello? Yeah okay. Alright. Okay, I will, thanks. [hangs up] It was negative.

Michael: Oh... God... [stomps] God! [throws hockey stick and yellow paper bracelet down] We're gonna b*at this, okay? We're gonna... come here [hugs Kevin].
Michael: Well, apparently in the medicine community, negative means good. Which makes absolutely no sense. In the real world community, that would... be... chaos.
Kevin: This is awesome. Thanks, you guys.
Michael: Okay, who's this from? Wowwwee, look at that! Jersey!

Dwight: Turn it around. Turn it around.

Michael: Cool. Ohh. Great. From Dwight.

Dwight: Number one!!

Michael: Thank you... Dwight. That's great. Thanks.

Pam: Michael?

Michael: Yeah.

Pam: This is from all of us.

Michael: Oh! You didn't need to do that. ...Nightswept. This is... really amazing. Thank you. I love it.
Pam: Michael's birthday was actually pretty cool. It was a good day. I don't know... It was a good day.
Dwight: Kevin Malone, you're next. Spit that out. [Kevin shoves the rest of the donut he's eating into his mouth] Spit... Okay, come on, let's go.
Jim: You look cute today, Dwight.
Dwight: Thanks, girl.
Jim: So, yesterday Dwight found half a joint in the parking lot. Which is unfortunate because as it turns out, Dwight finding drugs is more dangerous than most people using drugs.
Dwight: Let's go over some of the symptoms of marijuana use, shall we? You tell me who this sounds like: slow moving, inattentive, dull, constantly snacking, shows a lack of motivation.
Kevin: [nods] Hey...
Dwight: I like the people I work with generally, with four exceptions. But someone committed a crime and I did not become a Lackawanna County Volunteer Sheriff's deputy to make friends. And by the way, I haven't.
Jim: [mimicking Stanley] I enjoy the tangy zip of Miracle Whip.
Pam: [laughs] Jim does the best impressions. Sometimes he'll look up at me from his desk and he'll just be someone else. Like he'll go um, [makes mournful face, giggles] that's supposed to be Phyllis. I can't do it as good as he can.
Kelly: And the guys are saying, chug, chug, chug, but I'm so small and all I'd eaten that day was one of those Auntie Anne pretzels from the food court so I said "Is it okay if I sip it?" and they said no, but Ryan seemed cool either way.
Dwight: Stop! This is not Kelly Kapoor story hour. Illegal drugs were consumed on company property, okay? Your ass is on the line, mister! My ass is is on the line! Now I'm going to ask you again. What time did you go home last night?

Kelly: Six.
Dwight: I didn't know that you were at a party on Saturday night.
Ryan: I go to a lot of parties.

Dwight: Okay, I'm gonna need to search your car. Give me you keys.

Ryan: I am not giving you my keys.

Dwight: Don't make me do this the hard way.

Ryan: What's the hard way?

Dwight: I go down to the police station on my lunch break. I tell a police officer, I know several, what I suspect you may have in your car. He requests a hearing from a judge and obtains a search warrant, once he has said warrant, he will drive over here, and make you give him the keys to your car, and you will have to obey him.

Ryan: Yeah, let's do it that way.

Michael: Ry, is he bugging you? Dwight, dude, you gotta take a chill pill, man. It was one joint in the parking lot. You know, you're totally harshing the office mellow.

Dwight: I can't stop this investigation. It is my job.

Jim: Whoa. You are a volunteer.

Dwight: I volunteered for this job.

Jim: And that's not the same.

Dwight: It is my duty...

Jim: [interrupting] Volunteer duty.

Dwight: ...to investigate the crime scene. I have six more interviews to go and then I will reveal what I know.

Michael: [fake coughing] Narc!

Kevin: [giggling]

Dwight: If you are attempting to compliment me then you have done a very good job.

Michael: I wasn't attempting to compliment you.

Dwight: Well, you have...

Michael: Uuf, well...

Dwight: ...because being a narc is one of the hardest jobs that you can have...

Michael: [shakes head] Okay...

Dwight: ...and I am very proud of being a narc.

Michael: Why don't you just cool it, cool it Dwight, please, God! [to Jim] Dude, where's my office? [Jim quietly laughs] I totally lost it, 'cause I was half-baked. Smokin' doobies. Doobie brothers, I was smokin' doobies with my brothers. Peace out, Seacrest!

Jim: Well, your office is behind you.

Michael: Thanks. M-m-munchies. Who wants some munchies?
Ryan: I don't think Michael's ever done drugs. I don't know if anyone has ever offered him any.
Dwight: Oscar visited Mexico when he was five to attend his great-grandmother's funeral. What does that mean to a United States law enforcement officer? He's a potential drug mule.
Dwight: Have you ever taken any illegal drugs?
Oscar: No, I have not.

Dwight: Do you think it's possible that maybe you could have had some drugs in your system without you knowing about it?

Oscar: What are you implying?

Dwight: Have you ever... pooped... a balloon?

Oscar: Okay. I'm done with this.

Dwight: He sure left in a hurry.
Dwight: I don't want to blow this. This is what all good law enforcement officers dream of. The chance to solve an actual crime.
Dwight: Do you know what this is? [pushing a photo toward her]
Phyllis: Yes, it's marijuana.

Dwight: How do you know that?

Phyllis: It's labeled.

Dwight: [grabs pictures back and looks at it] Dammit.
Creed: That is Northern Lights Cannabis Indica.
Dwight: No, it's marijuana.
Jim: I'm just saying that you can't be sure that is wasn't you.
Dwight: That's ridiculous, of course it wasn't me.

Jim: Marijuana is a memory loss drug, so maybe you just don't remember.

Dwight: I would remember.

Jim: Well, how could you, if it just erased your memory?

Dwight: That's not how it works.

Jim: Now how do you know how it works?

Dwight: Knock it off, okay, I'm interviewing you.

Jim: No! You said that I'd be conducting the interview when I walked in here. Now exactly how much pot did you smoke?

Dwight: [opens eyes wide in total surprise]
Oscar: So Pam told me that you do a great Stanley impression, I'd love to hear it.
Jim: Oh, um...[mimicking Stanley] Why do you keep CC'ing me on things that have nothing to do with me? [Pam and Oscar laugh, Stanley walks in, and Oscar leaves quickly]

Stanley: Is that supposed to be me?

Jim: Oh, hey Stanley. Uh, I was just doing an impression.

Stanley: I do not think that is funny.

Pam: He does everyone in the office.

Stanley: Hmmmpt.

Pam/Jim: [in unison] I do not think that is funny.

Pam: Jinx! Buy me a coke.

Jim: Oh...

Pam: No, no, no, no talking. Jim is not allowed to talk until after he buys me a coke. Those are the rules of jinx, and they are unflinchingly rigid. [Jim puts money in drink machine, selection is sold out]

Pam: Sold out? That has never happened in the history of jinx.

Jim: [mouths] C'mon!

Pam: Sorry, that's not my problem.

Jim: [presses drink button, looks at camera, makes Jim-face]
Dwight: I know you're innocent, but I can't look like I'm treating you any differently.
Angela: I understand.

Dwight: Where were you yesterday after work?

Angela: [smiles knowingly]
Michael: Uh-oh. Uh-oh. Who's he calling? Ratting somebody out. Narc! Narc! Kevin?
Kevin: That is so good, Michael

Michael: Remember the narc bit? [laughs] Uh-oh, who's in trouble?
Dwight: Attention everyone. Drug testers are coming in a couple of hours to test everyone's urine.
Michael: Waa... what? What are you talking about?

Dwight: Company policy. If drugs are found on the premises there is a*t*matic drug testing conducted within twenty-four hours.

Oscar: Is that true, Toby?

Toby: Oh, when you sign your job application you agree to comply with random drug testing.
Michael: Two nights ago, I went to an Alicia Keys concert at the Montage Mountain Performing Arts Center. I scored these great aisle seats. Anyway, after the opening act this beautiful girl sits down next to me and I never get to meet girls with lip rings and she had one. I don't know exactly how this happened but one of her friends started passing around some stuff and they said it was clove cigarettes, and I'm sure that it was clove cigarettes. Everybody in the aisle was doing it.
Michael: Okay, attention everyone the drug testing has been cancelled. Instead, I will be going around to each of you and doing a visual inspection.
Dwight: No you can't do that.

Michael: I can do that, it is my office.

Dwight: No you cannot. It has to be official, and it has to be urine.

Michael: Hmmm. Ha. [under his breath] Alright. Great.
Dwight: Kevin, what prescription drugs are you taking, besides Rogaine?
Kevin: I'm not taking Rogaine.

Dwight: Angela, what about you?

Angela: I don't take any prescription drugs.

Dwight: You're not on anything?

Angela: [Gives Dwight a knowing look]

Dwight: Good.
Kelly: So the first time we went out to dinner, it was like, whatever, fine, but I was so nervous. So this time I wanted to be special, so I bought a new dress! [Jim hunches his shoulders and grins] One of those kinds that is kinda low cut at top to show something, but not everything. [Jim shakes his head no in agreement] I mean not everything, Jim. [Jim shakes his head in agreement] I promise, I'm not that kind of...
Pam: Hey guys, what's going on?

Kelly: We're having the best conversation. [Jim, eyes wide, shakes his head, no]

Pam: Oh, okay. Well, Michael wants to see everyone in the conference room. [Jim shakes his head. Seems relieved to be getting away from Kelly] But you know what? We have a few minutes so you guys should definitely finish up your conversation. [Jim nods his head as if to say, "Gee, thanks, Pam."]

Kelly: So, I was looking so hot...
Michael: It has come to my attention that some people here think that the use of drugs is something to laugh about.
Phyllis: We don't feel that way.

Angela: No, not at all.

Oscar: You were the one joking around calling Dwight a narc.

Michael: No, uh, no. That was a test. I was testing you, and you all failed. Miserably. When I said that Dwight was a narc, how many of you defended him? How many of you said, "Hey, you know what, he's right? What he's doing is protecting this office from the evils of drugs."

Dwight: Thank you, Michael.
Michael: I am ridiculously anti-drug. So anti-drug that I am above suspicion in any way that involves suspicion, or testing of any kind.
Michael: Drugs ruin lives people. Drugs destroy careers. Take Cheech and Chong everybody knows that Cheech and Chong are funny, but just imagine how funny they would be if they didn't smoke pot. I want everybody to take a look to their left. Now I want everybody to take a look to their right. One of those people will be dead from drug use at some point in their lives. This year more people will use cocaine than will read a book to their children.
Stanley: Where did you get these facts?

Michael: Are these facts scaring you, or are they not?

Stanley: They are not.

Michael: Do you think that smoking drugs is cool? Do you think that doing alcohol is cool?

Stanley: No, I don't. I have a glass of red wine with dinner about once a week. For the antioxidants.

Michael: Okay, enough, enough, enough! I have written down a list of illegal drugs. Take a gander. How many of these are you familiar with?

Toby: Hookah is not an illegal drug, Michael.

Michael: Yes it is.

Toby: No it's not. It's a type of pipe. You can fill it with tobacco, often mixed with fruit, or other flavors.

Michael: Okay, you know what, Toby? Pam, can you take this down? [Pam throws her hands up to indicate she has no notepad] In addition to Toby's urine being tested, I would like to test his blood and his hair.

Toby: You can't do that.

Michael: I can test anyone randomly, and I have chosen you randomly.

Toby: That's not random.

Michael: Okay, eeny, meenie, miney, moe, is random. Okay, you know what? I'm going to need a volunteer to select one of these words and tell us of something tragic that happened in either their lives, or the lives of a loved one. Yeah, Pam.

Pam: I know that Jim has an amazing story about a relative of his who got caught up in the world of drugs.

Michael: Really? [Jim shakes his head no]

Pam: Uh, hmmm.

Michael: Jim it's okay. You can t... [Jim looks at Pam and shakes his head, Pam looks at him and gestures for Jim to go up and tell his story] This would be a good place to let it out, Jim. These are people you can trust. These are people who care about you. [Jim shakes his head no] It's okay, just we will not judge you. We are here to not judge you. [Jim stands up] Oh, he's doing it, okay. [Jim looks at Pam] It's okay. [Jim pretends to try, looking teary eyed, shakes his head no, mouths "I can't" and sits back down. Pam is amazed.] Oh. Okay, are you sure? [Jim shakes his head. Kevin pats his shoulder] That looked like it was going to be good. Alright. [Pam nods in admiration at Jim] Okay, well.
Pam: Wow! He really pulled out the big g*n. Fake crying. Did not expect that.
Michael: The point I'm trying to make with all of this people is that I hate drugs. I hate them, and based upon what I have seen you all don't quite hate 'em as much as I do so you are going to have a drug test, and I am not.
Dwight: No, you will be tested.

Michael: Yes, I will not be.

Dwight: You will be. That is the law according to the rules.

Michael: Okay, well Dwight just know that I've been very busy today and I got a lot of work to do and I wasn't planning on going to the bathroom and I don't even know if anything is going to come out, okay? So good. Thank you.
Dwight: Hi, Linda. Dwight Schrute, Assistant Regional Manager. You might remember testing my urine a few years back when I was applying to be a volunteer sheriff's deputy.
Linda: We test a lot of urine.

Dwight: Mine was green.

Linda: Oh, right. How are you?

Dwight: I'm all better.
Michael: So I need you to do some work on the St. Andrews account. I need your urine. I need some filing done.
Dwight: What kind of filing?

Michael: Just forget it. Just the urine.

Dwight: That goes directly to the tester.

Michael: Just. I need your urine.

Dwight: Like in a cup?

Michael: Yes in a cup, we're not animals, Dwight.

Dwight: For what purpose?

Michael: It's none of your business.

Dwight: Then I refuse.

Michael: Okay. Alright. Just, I went to an Alicia Keys concert, over the weekend, and I think I may have gotten high accidentally by a girl with a lip ring.

Dwight: Are you serious?

Michael: I need clean urine for the lady.

Dwight: But that's illegal.

Michael: Don't think of it that way. It's like, urine goes all over the place. You know, there's no controlling it. It just... goes

Dwight: Not my urine.

Michael: A cup could find its way under the urine. It might be an accident. It happens.

Dwight: Were you forced to do drugs at this concert?

Michael: No, just look. Look. Just... just fill up the cup.
Angela: Do you want to give Michael your urine?
Dwight: I want him to have all the urine he needs.

Angela: You're not going to get my permission on this.

Dwight: I know that. Don't you think I know that?
Linda: Yeah, we do testing all over the country.
Ryan: Cool. Hey, are you guys hiring?

Linda: You want to work at the urinalysis lab?

Ryan: Yeah. Maybe.
Dwight: My father's name was Dwight Schrute. My grandfather's name was Dwight Schrute. His father's name Dwide Schrude. Amish. I loved my father very much. Every morning he'd wake up at dawn and make us biscuits with gravy. When I was little my dad and I played a lot of games together. My dad cheated a lot but I never busted him on it. I would have, except I didn't know about it. He didn't tell me till years later. I was shocked when I found out.
Pam: What? [Jim shakes his head] Did you want to tell me something? You look like you want to tell me something. [Jim shakes his head no] You look like you have something really important to say and you just can't for some reason. [Jim smiles] Come on, you can tell me. Jim, you can tell me anything. [Jim stops smiling and looks down. Pam wonders what that means]
Kevin: I'd like a magazine.
Linda: We just need urine, sir.

Kevin: I'd still like one.
Michael: Dwight. Well, I passed the test thanks to you and your untainted pee. Thank you very much.
Dwight: That's great.

Michael: What's wrong? Where's your costume?

Dwight: It's a uniform and I turned it in today when I tendered my resignation.

Michael: Why? Wha...

Dwight: I took an oath when I was sworn in and I broke that oath today.
Pam: [placing a coke can in front of Jim] Here. [Jim looks confused] Just buy it from me. I haven't talked to you in hours and it's been weird and I really want to know what the hell's going on with Dwight. [Pam scoots the coke can towards Jim. Jim pulls out his wallet and hands Pam a dollar. He gives the coke back to Pam]
Jim: Hi.

Pam: Hey.

Jim: How much time do you have left on your break?

Pam: Ten minutes.
Michael: Since you did such a good job with the investigation, I decided to pull a few strings. Call in a few favors. and I've decided to make you official security supervisor of the branch.
Dwight: Really?

Michael: Yes, sir.

Dwight: That's fantastic because I've always felt that the security here sucked .

Michael: So you wanna? Thanks.

Hank: Dwight K. Schrute, I hereby declare you an honorary voluntary corporal in charge of assisting all activities security.

Michael: Okay.

Hank: Here's your badge.

Dwight: Thank you, Michael. Very nice. Great. [To Hank] Can I have a g*n?

Hank: No, I don't have a g*n.

Dwight: Okay, I'll have to bring in my bow staff.

Hank: I don't think so.

Michael: Good.

Dwight: [salutes] Thank you, Michael.

Michael: No. Oh. Uh...[awkwardly salutes]

Dwight: I need to go over some details with you.

Michael: Alright. [to Hank] Well, Thank you.

Dwight: First of all, Hank, how many orange traffic cones do you have?

Hank: Two.

Dwight: Oh, God.
Jim: Wow. What a terrible day to not be able to talk. Dwight was literally carrying around his own urine and dressed like one of the Village People. Why does he do the things that he does for Michael? I just don't get it. What is he getting out of that relationship?
Kevin: So, uh... you found a band for your wedding yet?
Pam: No.

Kevin: 'Cause I'm in a band. We really rock.
Jim: Yeah, I mean it's inevitable. I definitely overhear some wedding preparation, but I'm fine with it. She hears me arranging my social life. And we both have to hear Dwight order deer urine over the Internet, so it evens out.
Phyllis: Oh I got the 'Save The Date'.
Pam: Yeah?

Phyllis: Yeah, pretty stationery.

Pam: Oh, thanks!

Angela: I didn't get mine yet.

Pam: Uh...
Pam: There are a few people I decided not to invite, and that might make things kind of awkward but ... it's my wedding. And I don't want anyone there who has called me a hussy.
Michael: Yes, thanks, Fantastic Sam's. Adult Cut Plus. Comes with a shampoo and blow dry. We're doing I.D. photos today. Gotta represent.
Phyllis: Uh, on or off?
I.D. Photographer: Off.

Phyllis: Okay... [removes glasses]

Dwight: Oh! What is on your face? Is that a disguise?

Phyllis: [leaving the room] Excuse me.

Dwight: Clown paint.
Dwight: I.D. badges are long overdue. Security in this office park is a joke. Last year, I came to work with my spud g*n in a duffel bag. I sat at my desk all day, with a r*fle that sh**t potatoes at 60 pounds per square inch. Can you imagine if I was deranged?
Michael: That's a nice tie.
Ryan: Thank you.

Michael: That is... who makes that?

Ryan: Um, I don't...

Michael: Do you mind if I wear that for the photo?

Ryan: Um... let's um, let's keep our clothes.
Oscar: It's like child abuse. I say, if Jesus saw that, he'd freak out! He'd freak out, Toby! I mean on some levels... it's... and I'm supposed to work there. I'm supposed...
Michael: [walking into the Conference room] What's the dealio?

Toby: Just letting Oscar vent a little. We would use the break room, but the photographer's back there today.

Michael: What's the problem?

Oscar: Angela!

Toby: It's just a little dispute over a poster in their workspace.

Oscar: Since Christmas.

Michael: So what, you're having a little spat. I forget, are you guys dating?

Oscar: No.

Toby: Michael, can I talk to you for a uh, second please?

Michael: Yes.
Toby: Here's how I usually handle this: all I do is listen.
Michael: Yeah?

Toby: These things just have a way of working themselves out.

Michael: Okay.

Toby: It's like if you write someone a letter, when you're really angry... they say to keep it in a drawer for a couple days. Then you just never end up sending it.

Michael: What do you know about conflict resolution? Your answer to everything is to get divorced. So...

Toby: Okay.

Michael: Okay... what?

Toby: That was the right decision for me and my marriage.

Michael: Yeah, well... that's not gonna fly here. Because in this office, it is till death do us part... assuming we don't get downsized. [leans over to Pam] Uh, Pam, will you get Angela and meet us in the conference room please?
Michael: [holding up a binder] A mediator's tool chest. Okay, well, before we get started, you should know that are 5 different styles of conflict.[in a comedic voice] My Shaolin temple style defeats your monkey style.
Angela: Can we go? I have a lot of work to do.

Michael: No. Okay, this is important. The first style is lose/lose.

Oscar: What's the next one?

Michael: Just hold on, please! Okay, if we do lose/lose, neither of you gets what you want. Do you understand? You... you would both lose. Now I need to ask you, do you want to pursue a lose/lose negotiation?

Angela: Can we just skip to whatever number 5 is - win/win or whatever?

Michael: Win/Win is number four and number five is win/win/win. The important difference here is with win/win/win, we all win. Me too. I win for having successfully mediated a conflict at work.
Michael: [in front of poster] Okay, since this is the disputed poster. Now, one at a time, I want you to express your feelings using "I" emotion language and no judging or "you" statements.
Angela: I got this poster for Christmas, and I feel I want to see it everyday. It makes me feel like the babies are the true artists, and God has a really cute sense of humor.

Michael: Come on, seriously, that?

Oscar: I don't like looking at it. It's creepy, and in bad taste, and it's just offensive to me. It makes me think of the horrible, frigid stage mothers who force the babies into it. It's kitsch. It's the opposite of art. It destroys art. It destroys souls. This is so much more offensive to me than hardcore p*rn. I'm talking about the...

Michael: Okay, okay. Stop, stop, stop! Let's see if we can just brainstorm and find some creative alternatives that are win/win.

Pam: Win!

Michael: Yes. Thank you, Pam. How about Angela makes the poster into a t-shirt, which Oscar wears. That way, he can never see it and whenever she looks at Oscar, she can see it. Win/win/win.

Oscar: No.

Angela: That's... no...

Michael: Okay... well, brainstorm. Own the solution.

Angela: How about, I leave it up?

Oscar: How 'bout, she takes it down?

Pam: How about, Angela can keep it up on Tuesdays and Thursdays?

Michael: Okay, that is called a compromise. And it is style 3. And it is not ideal. To sum up, win/win - make the poster into a t-shirt, win/lose - take the poster down, compromise - Tuesdays and Thursdays. And the answer is... make the poster into a t-shirt! Win/win.

Pam: Win.

Oscar: Fine.

Angela: But, it...

Michael: [claps his hands twice] It is done!

Pam: Win
Photographer: [snaps a photo of Creed, then Creed turns to the side for a profile sh*t] No, you're all good.
Creed: Great. [gets up and leaves]
Pam: Hey, Angela.[hands her a Save The Date card] I didn't have your zip code.
Angela: Oh. Thanks.
Angela: It was hand delivered. But, I did get a Save The Date after all. It's not my taste.
Toby: You solved it?
Michael: Yes.

Toby: Well, good. We can, uh, throw that one out. [shuffles through papers]

Michael: Are those all the other complains?

Toby: Mmm-Hm.

Michael: I would like to see those please.

Toby: I... I can't do that.

Michael: You can't do that, huh? Huh, ok. Now you and I have a conflict. I order you to give me that file.

Toby: That... [shakes head and places hand over the file]

Michael: Okay. [yanks the file away, despite Toby's resistance] There! No more conflict. [looks at the camera] I had to use win/lose on that. It was not pretty. [looks back at Toby] All right... is that it?

Toby: [sighs and pulls out a box under his desk] It's all Dwight's.
Toby: Every Friday at 4, I have a standing appointment with Dwight for him to file a grievance against Jim. I tell him that I'm sending them to a special file in New York. That box is the special file in New York.
Michael: Ohh... God. Alright. Why do I have to do everything?
Photographer: Are you sure? [looks at Oscar, who is front of the camera, holding the baby poster in front of his chest]
Dwight: Oh, he's sure. Just sh**t.

Photographer: [sh**t twice]
Michael: [looking through papers in the complaint box] This is from Kevin. He says Stanley uses his Miracle Whip without asking. Meredith complains that everyone talks too loud in the morning and the lights are too bright. Creed... huh. Duh, duh. Creed is sick of looking at the redhead all day and wants a seat facing the receptionist.
Pam: Nice.

Michael: You will notice that not one of these complaints is against me.
Toby: Actually, I have a separate folder for complaints against Michael. This [unlocks a drawer]is January through March of this year. [pulls out a fairly large folder]
Michael: How many of you have at one time gone to Toby to complain about another employee? [looks at all the employees, most of whom raise their hands] And... did you get what you wanted, or were you merely listened to, you forget about your problem, and you move on? [employees mumble "merely listen to and forgotten..." ] That is outrageous! I love this place... and it pains me to see all of the negativity festering... [sighs] Okay, today we are going to get everything out of these files and into the open, where it can be resolved. Alright, how about the Phyllis/Angela dispute?
Angela: You already did me.

Michael: That's what she said. [Jim mouths these words along with Michael] The thing is, Angela... you are in here an awful lot. You have complained about everybody in the office, except Dwight, which is odd because everyone else has had run ins with Dwight. Toby, by the way, what does "redacted" mean? There is a file full of complaints in here marked "redacted"... ?

Toby: Yeah, it just means whoever complained came to me later and withdrew it, so I took their name off.

Michael: Oh, ok. There were a bunch of complaints about Dwight that were put in the redacted file about six months ago.

Dwight: Whoa.. wha... wait. If someone has a problem with me, why would they withdraw it six months ago?

Pam: [notices Angela's intense concern] Um... let's move on. I volunteer. Did anyone have a problem with me?

Michael: All right, Pamela. Come on down! Let's do it! And [looks through the file]... okay. Just one complaint. Actually, it has been withdrawn. So that is no help to us. Next.

Pam: Wait, what did it say?

Michael: Uh... [reading]"Does she have to plan her wedding on office time? Couldn't she do that at home?" [Pam looks Angela an angry look] Who else? Why don't we just warm up first? Warm up emotionally, all right?
Pam: I have this kind of big secret about Angela. And I've been really nice to her... and I haven't told anyone. And what the hell?!
Michael: Here is a Kelly complaint: "Ryan never returns my calls." Ugh, join the club.
Ryan: My voicemail's really spotty... sometimes...

Kelly: I didn't file a complaint. I was just talking.

Toby: To your HR representative.

Kelly: To my friend, I thought. I want that withdrawn.

Toby: Fine, I'll take your name off. [looks at Michael] So no one will know.

Michael: [crumbles up the complaint paper] Makin' progress. [Jim raises his hand] Yes?

Jim: Dwight tried to kiss me.

Michael: What?!

Jim: And I didn't tell anyone because I'm not really sure how I feel about it.

Dwight: That is not true. Redact it. Redact it!

Jim: Well, I'm not actually making a formal complaint. I just really think we should talk about it.

Dwight: Okay, question. [looks at Toby] When a name is withdrawn, does that also apply to the permanent misbehavior file in New York?

Toby: Sure.
Michael: Stanley. [off camera]
Pam: [gets up and walks over to Angela, whispering to her] Hey. Thanks for ratting me out!

Michael: [still of camera] You got a lot of anger under there buddy. Come on start us out. Unleash it.

Anglea: I didn't do it! [Michael and Stanley continue to talk off camera, but it's inaudible]

Pam: I find that hard to believe... considering you have problems with every single person in this entire office except Bobblehead Joe.
Michael: Someone complained that the men's room is "whites only". Stanley, you know that's not true.
Stanley: I didn't say that.

Creed: Then why is there a picture of a white man on the door? [Michael, along with the camera, look at the standard stick man on the bathroom door.]

Michael: Okay, Phyllis. You complained that Angela's giving you dirty looks. And you tried to get off the party planning committee.

Phyllis: No, I never said any such thing. Angela and I are close.

Michael: And... also, Phyllis, Stanley says that you cry too much, and that bugs him.

Phyllis: Stanley and I are close, too.

Stanley: We sit close.

Michael: Oh... ok.
Toby: [sits down for his ID picture] Just take it. [flashes goes off, while he is standing back up again]
Pam: I can't believe Angela. I went against my better judgment, and I gave her a Save The Date. And now it turns out she complained about me to Toby.
Jim: Well, it was redacted. Look, if she wants an invite, maybe she's just trying to be friends.

Pam: Don't take her side.

Jim: [sighs] Well, what does Roy think about everything?

Pam: I don't know. I try not to bother him about this kind of stuff.

Jim: You mean your thoughts and feelings?

Pam: Yeah.

Jim: Yeah...
Phyllis: I know you keep saying it's your space, even though there's no assigned parking, but I keep forgetting.
Angela: Yes, that's the problem.

Phyllis: I guess so...

Michael: Okay, well... all settled, then.

Phyllis: [whispering to Angela] I don't like you.
Michael: OK, Ryan. You told Toby that Creed has a distinct old man smell?
Creed: I know exactly what he's talking about. I sprout mung beans on a damp paper towel in my desk drawer. Very nutritious. But they smell like death.
Michael: All right, Kevin. You are accused of making sexually suggestive remarks to Angela that made her feel uncomfortable. Solution: Angela, you are to make sexually suggestive remarks to Kevin that will make him uncomfortable.
Kevin: I accept your decision!
Jim: Hey... you know what, Dwight? Maybe we should get our photo I.D. taken together.
Dwight: That doesn't make any sense.

Jim: Well, it saves time, you know. 'Cause we could just meet in the parking lot every morning. Walk in together. Perfect.

Photographer: [to Dwight, who is sitting in front of the camera] Smile.

Dwight: No.
Dwight: I never smile if I can help it. Showing one's teeth is a submission signal in primates. When someone smiles at me, all I see is a chimpanzee begging for its life.
Jim: This came out really well. [picks up Dwight's I.D. from the laminating machine and hands it to Dwight] There you go.
Dwight: This is humongous. I am not a security thr*at...

Jim: Oh.

Dwight: And my middle name is 'Kurt', not 'Fart'.

Jim: What did I write?
Dwight: I have another complaint for Jim's permanent file.
Toby: Talk to Michael. I gave him the box.

Dwight: What box?!
Phyllis: But I didn't report your snoring-
Stanley: Wednesdays, tearful. Tuesdays... [Dwight walks in and rummages through his complaint box]

Michael: Uh. Dwight.

Dwight: Ah... agh... dgh... Ahh! No, no! Four years of malfeasance unreported. This cannot stand.

Michael: Okay! Calm down.

Dwight: No! You calm down! Who's side is Toby on? Who's side are you on?

Michael: Hey, hey!

Dwight: Him or me?

Michael: Stop.

Dwight: Him or me? I cannot work with Jim anymore.

Michael: Okay...

Dwight: Either he goes, or I go.

Michael: Dwight...

Dwight: You choose!

Michael: Stop...

Dwight: One of us is out of here by the end of today! [runs out]

Michael: Oh... kay...
Dwight: I am not bluffing!
Michael: Okay.

Dwight: Okay?

Michael: Yes.

Dwight: Do the right thing here, Michael. Okay, I have served you loyally for years.

Michael: Mm-hmm.

Dwight: I deserve this. You know I do!

Michael: [picks up Dwight's I.D. and snickers] You know your I.D. says you're a security thr*at?

Dwight: You have till five.
Dwight: Oh, look, Jim. There's a sales manager position open in Stamford. Want me to call Jan and tell her you're interested? I could put in a good word for you, 'cause I'll still be working here. Transfer! Transfer! Everybody! Transfer! Transfer! Transfer! Transfer!
Michael: Okay... you two, in the conference room with me. Nobody leaves until we work this out. Cage match!
Michael: Cage matches? Yeah, they work. How could they not work? If they didn't work, everybody would still be in the cage.
Michael: Okay, so Dwight, in your own words - [reads from complaint paper] "Someone replaced all my pens and pencils with crayons. I suspect Jim Halpert." [flips to another paper] "Everyone has called me 'Dwayne' all day. I think Jim Halpert paid them to."
Jim: [laughs] Yes! Five bucks each. And it was totally worth it.
Michael: [reading] "This morning, I found a bloody glove in my desk drawer and Jim Halpert tried to convince me I committed m*rder. I think he may be the real m*rder." [flips to another paper] "Jim Halpert said there was an abandoned infant in the woman's room. When I went to save the child, I saw Meredith on the can." Gah. "This morning, I knocked myself in the head with the phone."
Jim: That actually took a while. I had to put, uh, more and more nickels into his handset, till he got used to the weight, and then I just... took 'em all out.
Michael: [reading] "Every time I typed my name, it said 'Diapers'."
Jim: Just a simple macro. You know, these actually don't sound that funny one after another. But he does deserve it, though.
Michael: "By the end of the day, my desk was about two feet closer to the copier."
Jim: Yeah, I just moved it an inch every time he went to the bathroom. And that's how I spent my entire day that day.
Michael: The Japanese have this thing called shiatsu massage, where they dig into your body, very hard. And it is very painful. And apparently, some people throw up. But the next day they feel great. I've never had one. They sound awful.
Jim: Maybe Stanford would be cool.
Dwight: It's a good market. Higher volume.

Jim: Yeah. Maybe we should both go.

Dwight: I have a girlfriend...

Jim: Sure you do, Dwight. Sure.
Michael: Hey, there's like, 300 more of these. Let's get to them later.
Dwight: So, you going to transfer Jim or not?

Michael: Maybe, I haven't decided yet. Let's get to work.

Dwight: I want an answer by tomorrow.
Michael: Okay. Oh... actually, tomorrow's not good. How about later in the week?
Dwight: Fine.

Michael: Good. Okay.
Michael: Hey! Wait. How about a group picture while you're here?
Photographer: I can't. I only get reimbursed for the I.D. photos.

Michael: Well... that's... what, a computer camera, right?

Photographer: You mean digital?

Michael: It'll take like two seconds.

Photographer: 20 bucks.

Michael: Ugh... All right. Everybody, [looks around at the employees] come on. Group photo for the newsletter.

Stanley: You gotta be kiddin' me.

Michael: Come on, everybody.

Dwight: Come on, let's go. Creed, Kevin, Oscar... andale! Let's go.
Photographer: One, two, three... smile. [camera flashes, but no one smiles] Try to smile.
Michael: We resolved a lot today, everybody. Think happy thoughts.

Photographer: Alright, I'm just gonna take it on three... whether you smile or not. One, two, three. [camera flashes]

Michael: Good, let's check that out. [looks at preview screen] Ew, okay, all right. One more. We'll take one more.

Photographer: That'll be another 20.

Michael: What?

Pam: Angela, I want to talk to you about something.

Michael: [off camera] You just press the button.

Angela: What?

Jim: No, Pam.

Pam: [looks at to Jim] I am. [looks at Angela] It's about the Save The Date.

Jim: Pam, it wasn't her.

Pam: What?!

Jim: I'm the one who complained about you.

Jim: I... I didn't know that Toby was gonna write it down. [the camera flashes] I was just venting.

Michael: [off camera] Okay, good. Check that out.

Jim: You know, it was one day.

Michael: [off camera] That's terrible.

Jim: And I took it right back. It was like...

Pam: Okay.

Phyllis: Oh, dear.

Michael: [off camera] Let's pay Mr. Price Gouger. [rejoins the group, on camera] Okay... we can do this. Come on, everybody. All right. Here we go. [flash goes off before he sits down]
Michael: It was really hard getting a good picture of fifteen people. He would not give me a good discount. And eight tries added up.
Michael: [flashback the photo being taken] One, two..[flash goes off] Didn't say three, did I?
Michael: But, I'm sort of an expert at Photoshop, so it turned out fine in the end. When people work together, there is going to be conflict. You can't outrun your problems.
Jim: [on Pam's answering machine] Hey, Pam... it's Jim. Um, I have a doctor's appointment in the city. So I probably won't be in till the late afternoon. Just thought I'd let you know. Okay, bye. [camera shows Jim sitting on a waiting coach in another Dunder Mifflin office]
Female worker: Okay, Jan will see you now.

Jim: Oh, thanks.
Michael: And that is why the idea of a cage match is so universally appealing. But here's the thing about cage matches: sometimes you have to open the cage. And that is something Toby will never understand.
Michael: Tonight the Scranton Business Park is having Casino Night and we are converting our warehouse into a full-blown gambling hall. And I know it's illegal in Pennsylvania, but it's for charity. And I consider myself a great philanderer. It's just... It's nice to know at the end of the day, I can look in the mirror and say, "Michael, because of you, some little kid in the Congo has a belly full of rice this evening." Makes you feel good.
Jim: Excuse me. How long is the wait for a table for two?
Dwight: I would never, ever serve you. Not in a million, billion years.

Pam: It's a nice tux.

Dwight: I know. It belonged to my grandfather. He was buried in it, so family heirloom.
Roy: So, what's the deal? We gotta pay for our own drinks? That's lame.
Pam: Come on, it'll be fun, and besides, I'm a roulette expert.

Dwight: Impossible. Roulette is not a game of skill, it is a game of chance.

Jim: I can always kind of win at roulette.

Dwight: Oh, really? Mmm-hmm.

Jim: Yeah.

Dwight: How would you do that?

Jim: Mind control.

Dwight: [laughs] You can't be serious. Are you serious?

Jim: Ever since I was a little kid, like, eight or nine, I could sort of control things with my mind.

Dwight: I don't believe you. Continue.

Jim: It was just little thing, you know, like I could make something shake or I could make a marble fall off the counter. You know, just little things.

Dwight: [scoffs] That's ridiculous. You know what? Uh... Why don't you move that coat rack? Excuse me, everyone! Attention in the office, please. Jim is about to prove his telekinetic powers and he needs absolute silence. Go ahead.

Jim: Okay, I'll try. [The coat rack wobbles] [Pam holds up an umbrella handle to the camera in another scene]

Dwight: Oh, my God.
Michael: I try not to think of it as lagging behind. It's more of a David-and-Goliath thing.
Jan: [on phone] Yeah, but... Well, the fact of the matter is that your branch is currently number four of the five branches that I oversee.

Michael: Top 80 percent!

Jan: Michael?

Michael: Yeah?

Jan: You know that I'm very serious here.

Michael: Jan, listen, I promise that I will kick it up a notch. Bam!

Jan: What?

Michael: Emeril. Oh, actually, while I have you, not that I have you or have ever had you, but we're having our Casino Night tonight and I think everyone would love to see their fearless leader here.

Jan: I thought that you were their fearless leader.

Michael: I am, but you are the Eva Peron to my Cesar Chavez.

Jan: [laughs] I think you can handle it.

Michael: Oh, come on. Come on.

Jan: I think so, Michael...

Michael: You know, it'd be fun. I can hear it in your voice. You need a break.

Jan: Goodbye, Michael.
Michael: Jan and I understand each other. The romance thing is sort of on hold for the time being, but we've remained good friends. Good friends with privileges. Not now, some day.
Michael: Okay, everybody. Tonight's event is to benefit the Boy Scouts of America.
Oscar: Again? We do that every year.

Michael: Well, they need our money. They don't have cookies like the Girl Scouts.

Oscar: It'd be nice to do something for people who are actually suffering.

Michael: Well, Oscar, if you don't like it, then you should concentrate on winning. Because the person at the end of the evening with the highest chip count will receive $500 to donate to the charity of their choice. And they will get a mini-fridge compliments of Vance Refrigeration.

Dwight: Yes!

Michael: So get your charities in to Pam. I, for example, am playing for Comic Relief.

Jim: That doesn't exist anymore.

Michael: Comedy is very much alive, as are homeless people.

Pam: No, they stopped making that show.

Michael: Well, then, they need our money more than ever.

Angela: You have to pick an approved, non-profit organization.
Creed: There's a great soup kitchen in downtown Scranton. Delicious pea soup on Thursdays. I'll probably give the money to them.
Kevin: Something with animals. Or people.
Kelly: Kobe Bryant has a foundation, and he is so hot. And he gave his wife the biggest diamond ring. I know he didn't do it. ...Maybe he did it.
Angela: We are giving money that has been gambled. Why don't we just deal drugs or prostitute ourselves, and donate that money to charity?
Michael: Oh, and another fun thing. We, at the end of the night, are going to give the check to an actual group of Boy Scouts. Right, Toby? We're gonna...
Toby: Actually, I didn't think it was appropriate to invite children since it's... You know, there's gambling and alcohol, and it's in our dangerous warehouse and it's a school night... And, you know, Hooters is catering. You know, is that enough? Should I keep going?

Michael: Why are you the way that you are? Honestly, every time I try to do something fun or exciting, you make it not... that way. I hate so much about the things that you choose to be. Okay, you know what? I will not donate my winnings to Comic Relief, since apparently it doesn't exist. I am going to donate to Afghanistanis with AIDS.

Jim: I think you mean the aid to Afghanistan.

Michael: No, I mean Afghanistanis with AIDS.

Phyllis: Afghani.

Michael: What?

Phyllis: Afghani.

Michael: That's a dog.

Pam: No, that's Afghan.

Michael: That's a shawl.

Dwight: Wait, canine AIDS?

Michael: No. Humans with AIDS.

Creed: Who has AIDS?

Jim: Guys, the Afghanistananies.

Michael: Okay, you know what? No. No. AIDS is not funny. Believe me, I have tried.
Michael: There are certain topics that are off-limits to comedians, JFK, AIDS, the Holocaust. The Lincoln Assassination just recently became funny. "I need to see this play like I need a hole in the head." [laughs] And I hope to someday live in a world where a person could tell a hilarious AIDS joke. It's one of my dreams.
Jim: What are you doing?
Pam: Oh, nothing.

Jim: "Till Death Do Us Rock."

Pam: They're wedding bands.

Jim: Oh.

Pam: Roy was supposed to pick the band, but he's concentrating more on the bachelor party now.

Jim: Wait, wait, where you going? I mean, even if you don't hire a band, you still have to watch the bands. Pam, these are people who have never given up on their dreams. I have great respect for that. And, yes, they're all probably very bad and that will make me feel better about not having dreams.

Pam: There's a KISS cover band in here.

Jim: Let's do it.
Pam: I'm pretty happy these days. I'm getting married soon and I'm getting along with everybody at work.
Jim: Why did I talk to Jan about transferring? Well, you know... I have no future here.
Michael: I have already put down the deposit. Do you understand how a deposit works?
Darryl: Mike, I am not having fire-eaters in a paper warehouse.

Michael: It's Casino Night like Las Vegas. There are fire-eaters all over the place.

Darryl: Except my warehouse.

Michael: Well, actually, it's my warehouse.

Dwight: Actually, it's owned by Beakman Properties, and Dunder Mifflin is four years into a seven-year lease.

Michael: Why are you here?

Dwight: When Darryl was coming, you said you wanted me here for protection.

Michael: Not. I said, not that.

Darryl: We just have a lot of stuff down there that could be stolen.

Michael: That's ironic.

Darryl: What?

Michael: That you are afraid.

Darryl: Why? 'Cause I'm from the hood?

Michael: Dinkin' flicka.

Darryl: [sighs] Dinkin' flicka.
Darryl: I taught Mike some, uh, phrases to help with his interracial conversations. You know, stuff like, "Fleece it out." "Going mach five." "Dinkin' flicka." You know, things us Negroes say.
Michael: Give me some. [Michael and Darryl perform simultaneous hand gesture]
Darryl: Oh, yeah, I taught him a handshake, too.
Jim: [Jim ejects a videotape from the VCR and puts in a new one] Wow. I don't know how you're gonna decide. They are all extremely good.
Pam: I think I should hire them all. Do like Lollapalooza.

Jim: Yes.

Pam: Have three stages, yeah.

Jim: Your mom would love that. She would. Oh, this band is called Scrantonicity.

Pam: Oh.

Jim: Let's take a look. Nice.

Pam: Oh, wait. That's Kevin. On the drums.

Jim: What?

Pam: On the drums! On the drums!

Jim: Oh, my God, that's Kevin! Great song, Kev. Oh, my God, he's the drummer and the singer.
Kevin: We really don't do a lot of weddings. We actually don't play in public very often. We are all really hoping that Pam's wedding works out. This could be a turning point for the band.
Jim: Wow. Oh!
Pam: Oh, my...

Jim: Yeah, you haven't seen that since 1983. That is amazing. Okay, we have to sign him. I'm gonna call the label, we're gonna...

Pam: No! No!

Jim: No, Pam, you're gonna lose him to another wedding.

Pam: No, come back! No, no, no!

Jim: Kev!
Pam: Jim is great. Being with him just takes away all the stress of planning my wedding.
Michael: [phone rings] Yes
Pam: [phone rings] Michael, Carol Stills for you.

Michael: Who?

Pam: Carol Stills.

Michael: Do I know a Carol Stills?

Pam: Your realtor.

Michael: Oh, yeah, put her through. Hey Carol, how goes the real estate biz? Is it real good?

Pam: It's still me.
Pam: Sometimes I don't put Michael through until he's already said something. I look at it as a practice run for him. He usually does better on the second attempt.
Pam: Carol, you're on with Michael.
Carol: [on phone] Hello, Michael?

Michael: Hi, Carol. How you doing?

Carol: I'm great. I just needed one last signature for your mortgage insurance.

Michael: Oh, hey, no problemo. Incidentally, I love the place.

Carol: Oh, great.

Michael: Great. It has a little bit of a weird smell. It's okay. At Christmas, the tree helped.

Carol: Oh, good, I'm glad. Can I drop it over later?

Michael: Actually, I'm sort of hosting this charity thing in our warehouse, Casino Night.

Carol: Oh, great.

Michael: Yeah, it'll be good. You know what? Why don't you come by? Bring the papers, I'll sign them and then you can stay and have a drink.

Carol: To the casino thing?

Michael: Yeah. It'll be fun. What do you...[phone rings] What do you...

Carol: What?

Michael: Oh, I'm sorry. Could you hold on? Yes?

Pam: Michael, Jan's on line two.

Michael: Okay, put her through. [Deep voice] Jan Levinson, I presume?

Pam: It's still me. Uh, Jan, here's Michael.

Jan: Michael?

Michael: Hey, Jan. How you doing?

Jan: You know, I... I thought about it and you are right.

Michael: I am?

Jan: I could use a little fun. So, I am going to drive up for your Casino Night.

Michael: Oh, okay.

Jan: Incidentally, what is the charity?

Michael: AIDS.

Jan: Okay, then. I will see you tonight.

Michael: Okay, sounds great.

Jan: Bye-bye.

Michael: Bye Hello, Carol? Hi, sorry about that. I just...

Carol: No problemo.

Michael: Right.

Carol: To answer your question...

Michael: Yeah?

Carol: Yes.

Michael: What?

Carol: I'd love to go.

Michael: Okay.

Carol: I have to get a sitter, but that shouldn't be a problem.

Michael: Problem. Good.

Carol: And I'll bring the papers, too.

Michael: Good, All right. Sounds great.

Carol: I'll see you tonight.

Michael: Bye.

Carol: Bye.

Michael: Two queens on Casino Night. I am going to drop a deuce on everybody.
Pam: [People playing casino games as the actual Casino Night begins] Oh, my God!
Roy: Yeah! That's great.

Michael: Hey, hey.

Carol: Hi.

Michael: Hey, Carol.

Carol: Hi.

Michael: You look great.

Carol: Thanks. Thank you for inviting me. It looks so great in here.

Michael: Oh, well... Kiss. [Michael kisses her on the cheek, pauses and then kisses her on the other cheek] That's how we do it in the paper biz. It's European and... Yes? Ah, Dwight [Kisses cheeks]

Dwight: Code name Re/Max is here. No sign of Lan Jevinson.
Dwight: I'm Michael's wingman. I've got his back. Two dates. He's got two dates tonight. My job is to keep Jan away from Carol and vice versa. Michael said, "We must deceive them, so as not to hurt them, and in that way, we honor them."
Michael: Can I get you a drink? The food is from Hooters.
Carol: Drink would be good.

Michael: Okay.
Creed: Oh, I steal things all the time. It's just something I do. I stopped caring a long time ago. You should see how many supplies I've taken from this place. Honestly, I love stealing things.
Billy's Girlfriend: I'm gonna get a drink. Do you need anything?
Billy: No, I'm fine. Thank's sweetheart.

Billy's Girlfriend: Okay.

Michael: Billy, your nurse is hot.

Billy: That's my girlfriend.

Michael: Your nurse became your girlfriend? Sweet.

Billy: She was never my nurse. I met her at Chili's. She was my waitress.

Michael: Chili's is great.
Michael: Welkommen, Bienvenue, and welcome to Monte Carlo! Dwight. I am no longer your boss. Lady Fortune is your boss.
Stanley: [Under his breath] Will Lady Fortune give me a raise?

Michael: Shut it, shut it, shut it. Will Lady Fortune be your mistress? Only time will tell, my friends. Leave all your preconceived notions about casinos at the door. Old friends, new lovers, and the disabled! Welcome all! Great, okay. Shuffle up and deal. Let's get it started! Black-Eyed Crows.

Dealer: Okay, the game is No-Limit Texas Hold'em. Good luck, everybody. That's at least four red chips to you, sir.

Michael: All-in. [Other players fold their hands]
Michael: Bluffing is a key part of poker, which is too bad, because I'm not very good at bluffing. Did you believe me?
Toby: I'll call.
Michael: What are... That's insane.

Toby: I have good cards.

Michael: Well, Toby, I went all-in on the first hand, so doesn't that tell you that I might have good cards, too? So don't be stupid. Just take it back.

Dealer: No, I'm sorry. He can't, sir. He's gone all-in.

Michael: Okay, all right, whatever.

Dealer: Flip them.

Michael: You really screwed that up. [Michael leaves]

Meredith: Wow.
Toby: I don't really play cards, but I'm not gonna lie to you. It felt really good to take money from Michael. Gonna chase that feeling.
Dwight: I expect to do very well tonight. I have an acute ability to read people. Jim, for instance, has a huge tell. When he gets a good hand, he coughs.
Jim: [coughs] I will raise. [Dwight sighs and folds his cards] Thanks.
Jim: It's the weirdest thing. Every time I cough, he folds.
Carol: Wow, bad luck.
Michael: Yeah, whatever. Hey, you know what? If luck weren't involved, I would always be winning. [Sees Jan] Oh, my God. Oh, my God.

Jan: Michael?

Michael: Jan.

Jan: Hi.

Michael: Look, okay, I think we're all adults here, and it has always been my understanding that we have an open relationship.

Jan: What are you... Just... Wait, what're you talking about?

Carol: What does that mean?

Michael: After you said you weren't coming, I invited Carol to come and I don't think that I did anything wrong.

Jan: No. No, you didn't. Hi, I'm Jan. I'm Michael's boss.

Carol: Hi, hi.

Jan: Does anyone want a drink?

Carol: No, I'm good.

Jan: Okay. [Carol stares at Michael]

Michael: Um...

Dwight: Hey, hey.

Michael: Hey. What...

Dwight: Jan's here.
Dwight: Give me the dice.
Kevin: Come on, Dwight.

Dwight: Let's go.

Billy: It's all on you, baby. Let's go.

Angela: Good evening, Dwight. What is this?

Dwight: Evening, Angela. This is craps. I need to roll an eight. If I do, everyone wins.

Kevin: Yes.

Angela: Then roll an eight.

Dwight: Thank you, Angela.

Angela: Good luck, Dwight.

Dwight: Yeah! Yeah! [Kisses Angela, she slaps him and walks away smiling]

Kevin: Dwight, let's keep it going. Let's keep it going. Let's go.

Oscar: Let it ride. Let it all ride.

Dwight: Give me the dice!
Jim: Yeah, right.
Pam: "Yeah, right," what?

Jim: What was this? [Makes face]

Pam: [Laughs] I have good cards.

Jim: Really?

Pam: Mhmm, And I'm gonna take you all-in.

Jim: Wow. I think you're bluffing.

Kevin: Yeah, I think she's full of it.

Pam: Straight.

Jim: Oh. Three nines.

Kevin: Pam. Jim Halpert, ladies and gentlemen.

Jim: Thank you very much. It was fun.
Jan: Cosmopolitan, please.
Carol: Can I get a red wine? So, two hours? That's a long drive.

Jan: Well, it's part of the job, you know? Keep an eye on things. So... Why not? So, how long have you and Michael been...

Carol: Oh, well, actually, I guess this would be our first date. I guess.

Jan: Casino Night in the warehouse. Good sport.

Carol: Well, I'm having a nice time.

Jan: Oh, me too. Me too.
Ryan: One beer and one Seven and Seven with eight maraschino cherries, sugar on the rim, blended if you can.
Jim: So, that's still going on, huh? You and Kelly?
Michael: All right!
Dealer: The point is four. sh**t, roll it. Four!

Dwight: Come on, sh**t!

Michael: Four! [Holds dice in front of Carol] Blow. Blow for luck! Yeah! Also, you. Not playing favorites. [Holds dice in front of Jan] All right, here we go!

Carol: All right.

Michael: Yeah!

Dealer: Five.

Michael: So close. So close.

Dwight: Come on. [Turns to Jan] So where you staying? Radisson?

Jan: What?

Dwight: Super 8?

Jan: No, I...

Dwight: Motel 6? Best Western?

Jan: I didn't... I don't know...

Dwight: Holiday Inn? The Hyatt in Wilkes-Barre? You staying with Michael?
Kevin: I won the 2002 $2,500 No-Limit Deuce-to-Seven-Draw Tournament at the World Series of Poker in Vegas. So, yeah... I'm pretty good at poker.
Kevin: All-in.
Phyllis: Okay, let's do it.

Bob Vance: Good Luck, honey.

Phyllis: Oh, thank you, Bobby. But it doesn't matter, it's just fun to play.

Kevin: Three queens.

Dwight: Nice, very nice.

Phyllis: I have an ace.

Oscar: No, that's a flush.

Dwight: Oh, man!

Phyllis: Oh, I have a flush!

Bob: Yes!

Phyllis: Look, I won! Look I have all the clovers! You wanna play again?
Kevin: I suck.
Roy: She took you down, huh?
Kevin: I do not want to talk about it.

Roy: Hey, I saw your tape. Your band, Scrantonicity? You guys rock.

Kevin: Yeah?

Roy: Yeah, you guys wanna play our wedding?

Kevin: Awesome. Did Pam say it was okay?

Roy: Whatever. I'm in charge of the music.

Kevin: Dude, you will not be sorry.

Roy: Sweet. All right.

Kevin: All right.
Jan: Smoke?
Jim: No, thanks. You having fun?

Jan: Fabulous time. I drove two and a half hours to get here.

Jim: Yeah, we all really...

Jan: Left work early, drove down here. And I am completely underdressed

Jim: Well, I think you look great.

Jan: Why did I hook up with Michael?

Jim: Yeah, why did you?

Jan: It was very late, Jim. Very... Very late and... Have you given any more thought to the transfer?

Jim: Oh, yeah.

Jan: Good. Have you told anyone?

Jim: No.

Jan: Well, you should.
Bob: Excuse me. Big moment. The evening's chip leader and winner of this beautiful mini-refrigerator courtesy of Vance Refrigeration, Creed Bratton, Dunder Mifflin!
Creed: Thanks. I never owned a refrigerator.
Roy: Sorry, babe. I am just b*at.
Pam: It's okay. I'll see you at home.

Roy: Okay. Hey, don't try to lose too much money, all right?

Pam: Okay.

Roy: If you still want a honeymoon. Hey, Halpert. Keep an eye on her, all right?

Jim: Okay, will do.

Roy: See you.

Pam: Bye! Hey.

Jim: Hey, how's it going?

Pam: Good, especially after I took all your money in poker.

Jim: Yeah. Hey, can I talk to you about something?

Pam: About when you want to give me more of your money?

Jim: No, I...

Pam: Did you wanna do that now? We can go inside. I'm feeling kind of good tonight.

Jim: I was just... I'm in love with you.

Pam: What?

Jim: I'm really sorry if that's weird for you to hear, but I needed you to hear it. Probably not good timing, I know that. I just...

Pam: What are you doing? What do you expect me to say to that?

Jim: I just needed you to know. Once.

Pam: Well, I um... I... I can't.

Jim: Yeah.

Pam: You have no idea...

Jim: Don't do that.

Pam: ...what your friendship means to me.

Jim: Come on. I don't wanna do that. I wanna be more than that.

Pam: I can't. I'm really sorry if you misinterpreted things. It's probably my fault.

Jim: Not your fault. I'm sorry I misinterpreted our friendship.
Jan: Hey. I'm leaving.
Michael: Hey, okay.

Jan: So, I just wanted to congratulate you on a fantastic evening. You did the company proud.

Michael: Thank you.

Jan: And thanks for inviting me. You were right, I needed it. So, thanks.

Michael: Okay. Thanks for coming.

Jan: Nice to meet you.

Carol: You, too.

Jan: And you guys have a good time together.

Michael: Okay. Talk to you Monday.

Jan: Yeah.

Carol: Goodbye.

Michael: Good night. She's a good boss.

Carol: She seems really nice.

Michael: Oh, she's great.
Michael: Love triangle. Drama. All worked out in the end, though. The hero got the girl. Who saw that coming? I did. And Jan was really happy for me. So actually the hero got two girls. He got the girl that he works with and he got the girl that he buys real estate from. So, I've got my New York girl and my local flavor. Life is good.
Pam: [On phone] About 10 minutes ago. No, I didn't know what to say. Yes, I know. Um, I don't know, mom, he's my best friend. Yeah, he's great. Yeah, I think I am. [Jim enters] I have to go. I will. Listen, Jim... [They kiss]
"""
print(correct_format(transcript))

